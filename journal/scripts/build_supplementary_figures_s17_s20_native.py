#!/usr/bin/env python3
"""Redraw S17–S20 at one type scale from the existing frozen source tables.

No fitting or new inference is performed. S17 reuses the original violin and
schematic functions. S18 replays retained summary intervals; S19/S20 reproduce
the historical bootstrap seeds, draws, ordering and jitter exactly. The optional
companion validator compares plotted numerical artists with the original source
renderers, independently of the new layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_canvas import (COLORS, ERR_CAPSIZE, LW_DATA, LW_ERR, LW_HAIR,
                           LW_REF, MARKER_MS, PT_LEGEND, PT_SMALL, SEED_ALPHA,
                           SEED_MS, Margins, NativeCanvas)
from journal_style import label_color, style_axis, style_direct_color_labels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
OUT = ROOT / 'figures/supplementary'
FILENAMES = {17: 'figure_S17_panels_A-B.pdf', 18: 'figure_S18_panels_A-K.pdf',
             19: 'figure_S19_panels_A-I.pdf', 20: 'figure_S20_panels_A-B.pdf'}


def read(folder, name):
    return pd.read_csv(DATA / folder / name)


def build_s17():
    import build_alignment_animal_figure as source
    cv = NativeCanvas(240/72, 1, hgutter_pt=30,
                      margins=Margins(left=43, right=15, top=28, bottom=48))
    a = cv.panel('A', 0, 0, 6, title='Signed causal mapping', schematic=True)
    b = cv.panel('B', 0, 6, 6, title='Neuron-level residuals', grid='y')
    with patch.object(source, 'panel_title', lambda *args: None):
        source.animal_schematic(a)
        source.neuron_distributions(b, read('animal_learning_francioni', 'neuron_sd_residual_distributions.csv'))
    # The schematic is defined in these original normalized coordinates.
    a.set_ylim(-.02, 1.22)
    # Explicit bottom reserve retains both population labels below the ticks.
    cv.declare_reserve('B', bottom=14)
    return cv


def forest(ax, frame, names, labels, *, xcol='mean_difference',
           lowcol='ci95_low', highcol='ci95_high', factor=100, colors=None):
    part = frame.loc[names]
    y = np.arange(len(names))[::-1]
    means = factor*part[xcol].to_numpy(float)
    lows = factor*part[lowcol].to_numpy(float)
    highs = factor*part[highcol].to_numpy(float)
    ax.axvline(0, color=COLORS['mute'], lw=LW_HAIR, zorder=0)
    if colors is None:
        ax.errorbar(means, y, xerr=np.vstack([means-lows, highs-means]),
                    fmt='o', color=COLORS['shunting'], markeredgecolor='white',
                    markeredgewidth=.5, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, ms=MARKER_MS)
    else:
        for i, m, lo, hi, color in zip(y, means, lows, highs, colors):
            ax.errorbar(m, i, xerr=np.asarray([[m-lo], [hi-m]]), fmt='o',
                        color=color, markerfacecolor=color, markeredgecolor='white',
                        markeredgewidth=.5, markersize=MARKER_MS,
                        elinewidth=LW_ERR, capsize=ERR_CAPSIZE)
    ax.set_yticks(y, labels)
    ax.set_ylim(-.6, len(names)-.4)
    style_axis(ax, grid='x')


def build_s18():
    point = read('point_dendrite_credit_controls', 'condition_summary.csv')
    pc = read('point_dendrite_credit_controls', 'paired_contrasts.csv').set_index('contrast')
    dose = read('physical_alignment_dose', 'condition_summary.csv')
    dc = read('physical_alignment_dose', 'paired_contrasts.csv').set_index('estimand')
    rc = read('remaining_physical_experiments', 'paired_contrasts.csv').set_index('contrast')
    cv = NativeCanvas(480/72, 3, hgutter_pt=30, vgutter_pt=50,
                      margins=Margins(left=86, right=17, top=25, bottom=34))
    titles = ['Serial versus grouped star', 'Flexible point controls',
              'Alignment dose', 'Paired alignment-dose contrasts',
              'Aligned / reversed contrasts (H3)', 'Depth-by-alignment interactions (H2)']
    axes = [cv.panel(chr(65+i), i//2, (i%2)*6, 6, title=t)
            for i,t in enumerate(titles)]
    a,b,c,d,e,f = axes
    forest(a, pc, ['serial_minus_star__aligned__d1','serial_minus_star__aligned__d2',
                  'serial_minus_star__aligned__d3','serial_star_alignment_interaction__d3'],
           ['D1','D2','D3',r'D3 $\times$ alignment'])
    a.set_xlabel('serial − star (pp)')
    p = point[point.regime.eq('aligned') & ((point.architecture.eq('serial_tree') & point.credit.eq('full_bp') & point.depth.eq(3)) | point.architecture.str.startswith('point_mlp'))].copy()
    p['order'] = p.architecture.map({'point_mlp_active':0,'point_mlp_total':1,'serial_tree':2})
    p = p.sort_values('order')
    for i,row,color in zip(range(3),p.itertuples(),[COLORS['point_mlp']]*2+[COLORS['shunting']]):
        m,lo,hi = row.mean_test_accuracy,row.ci95_low_test_accuracy,row.ci95_high_test_accuracy
        b.errorbar([i],[m],yerr=np.asarray([[m-lo],[hi-m]]),fmt='o',color=color,
                   markeredgecolor='white',markeredgewidth=.5,elinewidth=LW_ERR,
                   capsize=ERR_CAPSIZE,ms=MARKER_MS)
    b.set_xticks(range(3),['active\nmatch','total\nmatch','serial\nD3'])
    b.set_xlim(-.7,2.7);b.set_ylim(.86,1.015);b.set_yticks([.90,.95,1.0]);b.set_ylabel('test accuracy')
    b.text(.98,.03,'y axis truncated',transform=b.transAxes,ha='right',va='bottom',fontsize=PT_SMALL,color=COLORS['mute'])
    style_axis(b,grid='y')
    colors={1:COLORS['mute'],2:COLORS['oracle'],3:COLORS['shunting']}
    for depth,marker in zip((1,2,3),('o','s','^')):
        p=dose[dose.depth.eq(depth)].sort_values('alignment_alpha')
        m=p.mean_test_accuracy.to_numpy();lo=p.ci95_low_test_accuracy.to_numpy();hi=p.ci95_high_test_accuracy.to_numpy()
        c.errorbar(p.alignment_alpha,m,yerr=np.vstack([m-lo,hi-m]),color=colors[depth],marker=marker,
                   ms=MARKER_MS,lw=LW_DATA,markeredgecolor='white',markeredgewidth=.5,elinewidth=LW_ERR,capsize=ERR_CAPSIZE)
        dy={1:-.021,2:.021,3:0}[depth]
        c.text(1.06,float(m[-1])+dy,f'D{depth}',ha='left',va='center',fontsize=PT_LEGEND,color=label_color(colors[depth]))
    c.set_xlim(-.07,1.23);c.set_xticks([0,.25,.5,.75,1]);c.set_ylim(.44,1.06);c.set_yticks([.5,.6,.7,.8,.9,1])
    c.set_xlabel(r'sensor alignment $\alpha$');c.set_ylabel('test accuracy');style_axis(c,grid='y')
    forest(d,dc,['alpha_0.75_minus_0.25','within_seed_linear_slope_per_unit_alpha'],
           [r'$\alpha$ 0.75 − 0.25','linear slope'])
    d.set_xlim(-2.5,30.5);d.set_xticks([0,10,20,30]);d.set_xlabel('depth-benefit change (pp)')
    forest(e,rc,['h3_serial_minus_grouped__aligned__d3','h3_serial_minus_grouped__rewired_tree__d3',
                 'h3_serial_grouped_alignment_interaction__d3','h3_star_minus_grouped__aligned__d3'],
           ['serial − point\naligned','serial − point\nreversed','alignment\ninteraction','star − point\naligned'],
           xcol='mean_pp',lowcol='ci_low_pp',highcol='ci_high_pp',factor=1,
           colors=[COLORS[k] for k in ('shunting','point_mlp','bp','oracle')])
    forest(f,rc,['h2_alignment_interaction__serial_bp','h2_alignment_interaction__grouped_bp',
                 'h2_alignment_interaction__shared_local','h2_alignment_interaction__path_local'],
           ['serial BP','grouped BP','shared\nLocalCA','path\nLocalCA'],
           xcol='mean_pp',lowcol='ci_low_pp',highcol='ci_high_pp',factor=1,
           colors=[COLORS[k] for k in ('shunting','point_mlp','local','pathway')])
    for ax in (e,f):
        ax.set_xlim(-2.5,35.5);ax.set_xlabel('paired difference (pp)')
    return cv


def build_s19():
    from analyze_prospective_learning_results import bootstrap_ci
    frame = read('trained_subtree_address','seed_outcomes.csv')
    names=['neuron_shared_k1','correct_subtree_k2','within_neuron_deranged_k2','random_dense_rank2','exact_transport','gated_point_emulation']
    labels=['neuron-shared','correct ancestry','route derangement','random rank-2','exact','gated point']
    colors=[COLORS[k] for k in ('low_rank','shunting','highlight','additive','bp','point_mlp')]
    cv=NativeCanvas(250/72,1,hgutter_pt=32,margins=Margins(left=95,right=17,top=27,bottom=38))
    a=cv.panel('A',0,0,6,title='Two-stream learning',grid='x')
    b=cv.panel('B',0,6,6,title='Context forgetting',grid='x')
    for ax,metric,count,seedbase in [(a,'test_accuracy',6,42000),(b,'context_switch_forgetting',4,43000)]:
        for index,(name,color) in enumerate(zip(names[:count],colors[:count])):
            values=frame[frame.condition.eq(name)][metric].to_numpy(float)
            assert len(values)==10
            ordered=np.sort(values)
            ax.scatter(ordered,index-.40+np.linspace(-.08,.08,len(values)),s=SEED_MS**2,color=color,alpha=SEED_ALPHA,edgecolors='none',zorder=3)
            m,lo,hi=bootstrap_ci(values,seed=seedbase+index)
            ax.errorbar(m,index,xerr=[[m-lo],[hi-m]],color=color,marker='D',markerfacecolor='white',
                        ms=MARKER_MS+1.2,lw=LW_ERR,elinewidth=LW_ERR,capsize=ERR_CAPSIZE,zorder=5)
        ax.set_yticks(range(count),labels[:count]);ax.set_ylim(count-.45,-.80)
    a.set_xlim(.10,.90);a.set_xticks([.2,.4,.6,.8]);a.set_xlabel('held-out accuracy')
    b.axvline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
    b.set_xlim(-.06,.68);b.set_xticks([0,.2,.4,.6]);b.set_xlabel('context-0 forgetting')
    cv.fig.text(.57,.045,'Small dots: seeds; diamonds: mean ± 95% CI',ha='center',fontsize=PT_SMALL,color=COLORS['mute'])
    return cv


def build_s20():
    from build_journal_figures import jitter,errorbar_mean,METHOD_COLORS
    seg=read('figure3','segment_metrics.csv')
    seg=seg[seg.E_count.gt(0)].copy()
    seg['depth_bin']=pd.cut(seg.topological_depth,bins=[-1,2,4,6,9,np.inf],labels=['0-2','3-4','5-6','7-9','10+'])
    cell_depth=seg.groupby(['root_id','depth_bin'],observed=True,as_index=False).credit_domain_fraction.mean()
    cv=NativeCanvas(215/72,1,hgutter_pt=30,margins=Margins(left=48,right=17,top=27,bottom=39))
    a=cv.panel('A',0,0,6,title='Route support')
    b=cv.panel('B',0,6,6,title='Direct-type capture (8 cells)')
    groups=[g.credit_domain_fraction.to_numpy(float) for _,g in cell_depth.groupby('depth_bin',observed=True)]
    for i,arr in enumerate(groups):
        a.scatter(i+jitter(arr.size,210+i,.07),arr,s=SEED_MS**2,color=COLORS['shunting'],alpha=SEED_ALPHA,edgecolor='white',linewidth=.2)
        errorbar_mean(a,i,arr,COLORS['shunting'],seed=220+i)
    a.set_yticks([0,.1,.2]);a.set_xticks(range(5),['0–2','3–4','5–6','7–9','10+']);a.set_xlabel('topological depth');a.set_ylabel('domain fraction')
    typed=read('figure3','typed_only_compression_curves.csv')
    typed=typed[typed.channels.eq(8)].copy();typed['credit_capture']=1-typed.residual.astype(float)**2
    for i,method in enumerate(['morphology-aware paths','random paths','depth-only bins','shuffled ancestry']):
        arr=typed[typed.method.eq(method)].groupby('root_id').credit_capture.mean().to_numpy(float)
        b.scatter(i+jitter(arr.size,370+i,.05),arr,s=SEED_MS**2,color=METHOD_COLORS[method],alpha=.60,edgecolor='white',linewidth=.2)
        errorbar_mean(b,i,arr,METHOD_COLORS[method],seed=380+i)
    b.set_xticks(range(4),['ancestry','random','depth','shuffle']);b.set_ylim(0,.92);b.set_ylabel('field capture')
    return cv


def build(numbers=(17,18,19,20), *, save=True):
    canvases={n:globals()[f'build_s{n}']() for n in numbers}
    if save:
        for n,cv in canvases.items():
            style_direct_color_labels(cv.fig)
            cv.save(OUT/FILENAMES[n],name=Path(FILENAMES[n]).stem,dpi=200)
    return canvases


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',type=int,nargs='+',default=[17,18,19,20],choices=[17,18,19,20])
    args=parser.parse_args();build(args.figures)
