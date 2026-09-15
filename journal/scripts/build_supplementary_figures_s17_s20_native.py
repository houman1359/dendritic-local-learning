#!/usr/bin/env python3
"""Redraw S17–S20 at one type scale from the existing frozen source tables.

No fitting or new inference is performed. S17 reuses the original violin and
schematic functions and adds the six paired animal contrasts. S18 replays
retained summary intervals with the paired seeds behind them; S19/S20 reproduce
the historical bootstrap seeds, draws and jitter exactly (S19 A regroups its
rows on a broken axis but keys every bootstrap draw by condition). The optional
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
                           LW_EDGE, LW_REF, MARKER_MS, ORDINAL_RAMP,
                           PT_LEGEND, PT_SMALL, PT_TICK, SEED_ALPHA, SEED_MS,
                           Margins, NativeCanvas)
from journal_style import label_color, style_axis, style_direct_color_labels

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
OUT = ROOT / 'figures/supplementary'
FILENAMES = {17: 'figure_S17_panels_A-B.pdf', 18: 'figure_S18_panels_A-K.pdf',
             19: 'figure_S19_panels_A-I.pdf', 20: 'figure_S20_panels_A-B.pdf'}


def read(folder, name):
    return pd.read_csv(DATA / folder / name)


def _neuron_violins(ax, neuron, *, pplus, pminus, ylim=(-1.7, 1.5)):
    """Neuron-level residual violins (the original data logic) on an axis
    clipped to the body of the distributions; every neuron beyond the limits
    is counted at an arrow-capped tick, and the two condition words are
    spelled out with a spanning rule under each population pair."""
    from scipy import stats
    order=[('P+','error increase'),('P+','error reduction'),
           ('P-','error increase'),('P-','error reduction')]
    positions=[0.0,1.0,2.5,3.5];colors=[pplus,pplus,pminus,pminus]
    ylo,yhi=ylim
    for (population,epoch),color,pos in zip(order,colors,positions):
        values=neuron.loc[(neuron.population==population)&(neuron.epoch==epoch),
                          'sd_residual_z'].to_numpy(float)
        parts=ax.violinplot(values,positions=[pos],widths=.72,showmeans=False,
                            showmedians=False,showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor(color);body.set_edgecolor(color);body.set_alpha(.28)
        mean=float(np.mean(values));sem=float(stats.sem(values))
        ax.errorbar(pos,mean,yerr=sem,fmt='o',ms=MARKER_MS,color=color,mec='white',
                    mew=.4,lw=LW_ERR,capsize=ERR_CAPSIZE,zorder=4)
        for tail,edge,sign in ((values[values>yhi],yhi,1),(values[values<ylo],ylo,-1)):
            if not len(tail):continue
            extreme=tail.max() if sign>0 else tail.min()
            ax.annotate('',xy=(pos,edge),xytext=(pos,edge-sign*.24),
                        arrowprops=dict(arrowstyle='-|>',color=color,lw=LW_ERR,
                                        mutation_scale=6,shrinkA=0,shrinkB=0),
                        annotation_clip=False,zorder=5)
            ax.text(pos+.16,edge-sign*.14,f'{len(tail)} at {extreme:+.2f}',ha='left',
                    va='center',fontsize=PT_SMALL,color=label_color(color),zorder=5)
    ax.axhline(0,color=COLORS['mute'],lw=LW_REF,ls='--',zorder=0)
    ax.set_ylim(ylo,yhi);ax.set_yticks([-1.5,-1,-.5,0,.5,1,1.5])
    ax.set_xticks(positions,['error\nincrease','error\nreduction']*2)
    ax.set_xlim(-.65,4.15)
    # A short rule spans each pair, and the population name sits on it.
    for x0,x1,group,color in ((-.36,1.36,'P+',pplus),(2.14,3.86,'P−',pminus)):
        ax.plot([x0,x1],[-.215,-.215],transform=ax.get_xaxis_transform(),color=color,
                lw=LW_EDGE,clip_on=False,solid_capstyle='butt')
        ax.text((x0+x1)/2,-.245,group+' trials',transform=ax.get_xaxis_transform(),ha='center',
                va='top',fontsize=PT_TICK,color=label_color(color))
    ax.set_ylabel('residual (z-score)')


def _animal_pairs(ax, animal, summary, *, pplus, pminus):
    """The six paired animal contrasts that the sign test is run on: every
    animal's P+ and P− contrast joined, means with the animal-bootstrap 95%
    interval from summary.json, zero rule."""
    level=summary['animal_level']
    for row in animal.itertuples(index=False):
        ax.plot([0,1],[row.pplus_contrast,row.pminus_contrast],color=COLORS['mute'],
                lw=LW_HAIR,alpha=.55,zorder=1)
    for x,col,color in ((0,'pplus_contrast',pplus),(1,'pminus_contrast',pminus)):
        values=animal[col].to_numpy(float)
        assert len(values)==6
        stat=level[col];lo,hi=stat['bootstrap_95_ci']
        assert abs(float(values.mean())-stat['mean'])<1e-9
        ax.plot(np.full(6,x)+np.linspace(-.05,.05,6),np.sort(values),linestyle='none',
                marker='o',markersize=SEED_MS+.6,markerfacecolor=color,
                markeredgecolor='white',markeredgewidth=.3,zorder=3)
        ax.errorbar(x+.22,stat['mean'],yerr=[[stat['mean']-lo],[hi-stat['mean']]],
                    fmt='D',ms=MARKER_MS,color=color,markerfacecolor='white',
                    markeredgewidth=LW_ERR,lw=LW_ERR,capsize=ERR_CAPSIZE,zorder=4)
    ax.axhline(0,color=COLORS['mute'],lw=LW_REF,ls='--',zorder=0)
    # The sign counts sit under the population names, off the data field.
    ax.set_xticks([0,1],[f"P+\n{level['pplus_contrast']['ordering_positive']}/6 above 0",
                         f"P−\n{level['pminus_contrast']['ordering_negative']}/6 below 0"])
    ax.set_xlim(-.45,1.55)
    ax.set_ylim(-.30,.24);ax.set_yticks([-.2,-.1,0,.1,.2])
    ax.set_ylabel('contrast (reduction − increase)')


def build_s17():
    import json
    import build_alignment_animal_figure as source
    pplus,pminus=source.PPLUS,source.PMINUS
    cv = NativeCanvas(200/72, 1, hgutter_pt=30,
                      margins=Margins(left=14, right=15, top=24, bottom=24))
    a = cv.panel('A', 0, 0, 3, title='Signed causal mapping', schematic=True)
    b = cv.panel('B', 0, 3, 5, title='Neuron-level residuals', grid='y')
    c = cv.panel('C', 0, 8, 4, title='Six-animal contrasts', grid='y')
    with patch.object(source, 'panel_title', lambda *args: None):
        source.animal_schematic(a)
    # The predicted-contrast baseline is named as the quantity panel C draws.
    for artist in a.texts:
        if artist.get_text()=='predicted contrast':
            artist.set_text('predicted contrast\n(reduction − increase)')
    _neuron_violins(b, read('animal_learning_francioni','neuron_sd_residual_distributions.csv'),
                    pplus=pplus,pminus=pminus)
    _animal_pairs(c, read('animal_learning_francioni','animal_signed_contrasts.csv'),
                  json.loads((DATA/'animal_learning_francioni'/'summary.json').read_text()),
                  pplus=pplus,pminus=pminus)
    # Explicit bottom reserve retains the population rule and names below the ticks.
    cv.declare_reserve('B', bottom=22)
    # The schematic is defined in normalized coordinates (x 0-1, y up to
    # 1.22).  Keep its proportions without letting an equal-aspect box
    # shrink the panel out of the row: lock the grid first, then widen the
    # x range to the locked box so the drawing fills the slot height.
    cv.lock_reserves()
    box=a.get_position()
    span=0.905                             # the drawing is 0.9 wide (x 0.055-0.95)
    rng=span*(box.height*cv.height_pt)/(box.width*cv.width_pt)
    assert rng>=1.20, rng                  # the drawing is 1.14 tall (y 0.03-1.165)
    a.set_aspect('auto');a.set_xlim(.5-span/2,.5+span/2);a.set_ylim(.60-rng/2,.60+rng/2)
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


def _seed_forest(host, segments, rows, *, xlabel, ticks, reference=0.0,
                 seed_jitter=.16):
    """Category-vs-value rows on a broken x axis with the seed fan drawn.

    ``rows`` are top row first: ``label``, ``mean``, ``lo``, ``hi``,
    ``seeds`` (all ten paired values, always drawn), ``color`` and
    ``hollow`` (an open mean marker for a row that is derived from the rows
    above it rather than a new comparison).  Widths of the segments are
    proportional to their data ranges, so one pp is the same length in every
    segment of the panel.
    """
    segs=_broken_x_segments(host,segments)
    def seg_for(x):
        return next(s for s in segs if s.get_xlim()[0]<=x<=s.get_xlim()[1])
    for index,row in enumerate(rows):
        color=COLORS.get(row.get('color','ink'),row.get('color'))
        seeds=np.sort(np.asarray(row['seeds'],float))
        assert len(seeds)==10
        ys=index+np.linspace(-seed_jitter,seed_jitter,len(seeds))
        for s in segs:
            s.plot(seeds,ys,linestyle='none',marker='o',markersize=SEED_MS,
                   markerfacecolor=color,markeredgecolor='none',alpha=SEED_ALPHA,
                   zorder=2,clip_on=True)
        m,lo,hi=row['mean'],row['lo'],row['hi']
        s=seg_for(m)
        s.plot([lo,hi],[index,index],color=color,lw=LW_ERR,zorder=3,solid_capstyle='butt')
        for xb in (lo,hi):
            s.plot([xb,xb],[index-.13,index+.13],color=color,lw=LW_ERR,zorder=3,solid_capstyle='butt')
        hollow=row.get('hollow',False)
        s.plot([m],[index],linestyle='none',marker='o',markersize=MARKER_MS,
               markerfacecolor='white' if hollow else color,
               markeredgecolor=color if hollow else 'white',
               markeredgewidth=LW_ERR if hollow else LW_HAIR,zorder=4)
    for s in segs:
        s.set_ylim(len(rows)-.45,-.55)
    if reference is not None:
        seg_for(reference).axvline(reference,color=COLORS['mute'],lw=LW_HAIR,zorder=0)
    segs[0].set_yticks(range(len(rows)),[r['label'] for r in rows])
    for s,t in zip(segs,ticks):
        s.set_xticks(t)
    # The x label belongs to the whole broken axis, so it is centred on the host.
    host.set_xlabel(xlabel)
    host.xaxis.set_label_coords(.5,-.16)
    return segs


def _paired_seed_diff(frame,left,right):
    """Per-seed left-minus-right test accuracy, in pp, paired by seed."""
    def pick(spec):
        f=frame
        for k,v in spec.items():f=f[f[k].eq(v)]
        return f.set_index('seed').test_accuracy
    diff=(pick(left)-pick(right)).dropna()
    assert len(diff)==10
    return 100*diff


def build_s18():
    point = read('point_dendrite_credit_controls', 'condition_summary.csv')
    pc = read('point_dendrite_credit_controls', 'paired_contrasts.csv').set_index('contrast')
    ps = read('point_dendrite_credit_controls', 'combined_seed_outcomes.csv')
    dose = read('physical_alignment_dose', 'condition_summary.csv')
    dc = read('physical_alignment_dose', 'paired_contrasts.csv').set_index('estimand')
    dseed = read('physical_alignment_dose', 'combined_seed_outcomes.csv')
    rc = read('remaining_physical_experiments', 'paired_contrasts.csv').set_index('contrast')
    cv = NativeCanvas(480/72, 3, hgutter_pt=30, vgutter_pt=50,
                      margins=Margins(left=20, right=17, top=25, bottom=34))
    titles = ['Serial versus grouped star', 'Flexible point controls',
              'Alignment dose', 'Depth benefit versus alignment',
              'Aligned and reversed placement', 'Depth-by-placement interaction']
    axes = [cv.panel(chr(65+i), i//2, (i%2)*6, 6, title=t)
            for i,t in enumerate(titles)]
    a,b,c,d,e,f = axes
    # One label gutter for every panel of both columns, so the two columns
    # lock to one axes width instead of the forest column going narrow.
    for ax in axes:
        cv.declare_reserve(ax, left=64, right=14)

    def row(name,label,seeds,*,color='ink',hollow=False,factor=100):
        r=pc.loc[name] if name in pc.index else None
        return dict(label=label,mean=factor*r.mean_difference,lo=factor*r.ci95_low,
                    hi=factor*r.ci95_high,seeds=seeds,color=color,hollow=hollow)
    # A: serial tree minus the resource-identical star, paired by seed; the
    # last row is the aligned-minus-reversed difference of the D3 row (derived).
    star_rows=[]
    for depth in (1,2,3):
        seeds=_paired_seed_diff(ps,dict(architecture='serial_tree',regime='aligned',credit='full_bp',depth=depth),
                                   dict(architecture='all_active_star',regime='aligned',credit='full_bp',depth=depth))
        assert abs(seeds.mean()-100*pc.loc[f'serial_minus_star__aligned__d{depth}'].mean_difference)<1e-6
        star_rows.append(row(f'serial_minus_star__aligned__d{depth}',f'D{depth}',seeds.to_numpy()))
    reversed_d3=_paired_seed_diff(ps,dict(architecture='serial_tree',regime='rewired_tree',credit='full_bp',depth=3),
                                    dict(architecture='all_active_star',regime='rewired_tree',credit='full_bp',depth=3))
    inter=(_paired_seed_diff(ps,dict(architecture='serial_tree',regime='aligned',credit='full_bp',depth=3),
                             dict(architecture='all_active_star',regime='aligned',credit='full_bp',depth=3))-reversed_d3).dropna()
    assert abs(inter.mean()-100*pc.loc['serial_star_alignment_interaction__d3'].mean_difference)<1e-6
    star_rows.append(row('serial_star_alignment_interaction__d3','D3, aligned −\nreversed',inter.to_numpy(),hollow=True))
    _seed_forest(a,[(-1.2,9.0),(28.5,33.5)],star_rows,xlabel='serial − grouped star (pp)',
                 ticks=[[0,5],[30]])
    # B: flexible point controls with every seed drawn; the serial D3 mark is
    # the same exact-BP run as the ladder's serial BP D3.
    p = point[point.regime.eq('aligned') & ((point.architecture.eq('serial_tree') & point.credit.eq('full_bp') & point.depth.eq(3)) | point.architecture.str.startswith('point_mlp'))].copy()
    p['order'] = p.architecture.map({'point_mlp_active':0,'point_mlp_total':1,'serial_tree':2})
    p = p.sort_values('order')
    for i,prow,color in zip(range(3),p.itertuples(),[COLORS['point_mlp']]*2+[COLORS['ink']]):
        seeds=ps[ps.architecture.eq(prow.architecture)&ps.regime.eq('aligned')&ps.credit.eq('full_bp')&ps.depth.eq(prow.depth)].test_accuracy.to_numpy(float)
        assert len(seeds)==10 and abs(seeds.mean()-prow.mean_test_accuracy)<1e-6
        b.plot(i+np.linspace(-.16,.16,10),np.sort(seeds),linestyle='none',marker='o',markersize=SEED_MS,
               markerfacecolor=color,markeredgecolor='none',alpha=SEED_ALPHA,zorder=2)
        m,lo,hi = prow.mean_test_accuracy,prow.ci95_low_test_accuracy,prow.ci95_high_test_accuracy
        b.errorbar([i],[m],yerr=np.asarray([[m-lo],[hi-m]]),fmt='o',color=color,
                   markeredgecolor='white',markeredgewidth=.5,elinewidth=LW_ERR,
                   capsize=ERR_CAPSIZE,ms=MARKER_MS,zorder=4)
    b.set_xticks(range(3),['active\nmatch','total\nmatch','serial\nD3'])
    b.set_xlim(-.7,2.7);b.set_ylim(.895,1.005);b.set_yticks([.90,.95,1.0]);b.set_ylabel('test accuracy')
    style_axis(b,grid='y')
    # C: the depth ladder is ordinal, so it takes the ordinal ramp, not arm hues.
    colors={1:ORDINAL_RAMP[1],2:ORDINAL_RAMP[2],3:ORDINAL_RAMP[3]}
    for depth,marker in zip((1,2,3),('o','s','^')):
        p=dose[dose.depth.eq(depth)].sort_values('alignment_alpha')
        m=p.mean_test_accuracy.to_numpy();lo=p.ci95_low_test_accuracy.to_numpy();hi=p.ci95_high_test_accuracy.to_numpy()
        c.errorbar(p.alignment_alpha,m,yerr=np.vstack([m-lo,hi-m]),color=colors[depth],marker=marker,
                   ms=MARKER_MS,lw=LW_DATA,markeredgecolor='white',markeredgewidth=.5,elinewidth=LW_ERR,capsize=ERR_CAPSIZE)
        dy={1:-.021,2:.021,3:0}[depth]
        c.text(1.06,float(m[-1])+dy,f'D{depth}',ha='left',va='center',fontsize=PT_LEGEND,color=label_color(colors[depth]))
    c.set_xlim(-.07,1.23);c.set_xticks([0,.25,.5,.75,1]);c.set_ylim(.55,.95);c.set_yticks([.6,.7,.8,.9])
    c.set_xlabel(r'task–sensor alignment $\alpha$');c.set_ylabel('test accuracy');style_axis(c,grid='y')
    # D: the D3-minus-D1 benefit at every dose, paired by seed (the two
    # summary contrasts of the old panel, 0.75 - 0.25 and the per-unit slope,
    # are stated in the caption; a pp difference and a pp-per-unit slope do
    # not share an axis).
    wide=dseed.pivot_table(index=['alignment_alpha','seed'],columns='depth',values='test_accuracy')
    dose_rows=[]
    for alpha in (0.0,.25,.5,.75,1.0):
        seeds=100*(wide.loc[alpha][3]-wide.loc[alpha][1]).dropna()
        r=dc.loc[f'depth_effect_alpha_{alpha:.2f}']
        assert len(seeds)==10 and abs(seeds.mean()-100*r.mean_difference)<1e-6
        dose_rows.append(dict(label=rf'$\alpha$ = {alpha:g}',mean=100*r.mean_difference,lo=100*r.ci95_low,
                              hi=100*r.ci95_high,seeds=seeds.to_numpy(),color='ink'))
    _seed_forest(d,[(-1.2,4.5),(28.5,33.5)],dose_rows,xlabel='D3 − D1 accuracy (pp)',
                 ticks=[[0,2,4],[30]])
    # E, F: one broken axis for both (identical segments, 5 pp ticks), so
    # equal differences land at equal x in the two panels.
    def rrow(name,label,color='ink',hollow=False):
        r=rc.loc[name];seeds=np.array([float(v) for v in r.seed_values_pp.split(';')])
        assert abs(seeds.mean()-r.mean_pp)<1e-6
        return dict(label=label,mean=r.mean_pp,lo=r.ci_low_pp,hi=r.ci_high_pp,seeds=seeds,color=color,hollow=hollow)
    e_rows=[rrow('h3_serial_minus_grouped__aligned__d3','serial − point\naligned'),
            rrow('h3_serial_minus_grouped__rewired_tree__d3','serial − point\nreversed'),
            rrow('h3_serial_grouped_alignment_interaction__d3','aligned − reversed\n(derived)',hollow=True),
            rrow('h3_star_minus_grouped__aligned__d3','star − point\naligned')]
    # S18F is published as S24B: use the same arm hues as S22B.
    f_rows=[rrow('h2_alignment_interaction__serial_bp','serial BP','ink'),
            rrow('h2_alignment_interaction__grouped_bp','grouped BP','point_mlp'),
            rrow('h2_alignment_interaction__shared_local','shared\nLocalCA','local'),
            rrow('h2_alignment_interaction__path_local','path\nLocalCA','bp')]
    segments=[(-1.6,1.6),(18,35.5)];ticks=[[0],[20,25,30,35]]
    _seed_forest(e,segments,e_rows,xlabel='paired difference (pp)',ticks=ticks)
    _seed_forest(f,segments,f_rows,xlabel='paired difference (pp)',ticks=ticks)
    return cv


def _broken_x_segments(ax, segments, *, gap_frac=0.035):
    """Split one panel box into abutting x-axis segments (an axis break).

    The host panel keeps its slot, title and manifest record; each returned
    inset owns one contiguous data range, so interior spans that hold no
    point are removed from the page instead of taking three quarters of the
    width. Widths are proportional to the data ranges, so every segment
    keeps the same scale.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]);ax.set_yticks([]);ax.grid(False)
    spans=[hi-lo for lo,hi in segments]
    usable=1.0-gap_frac*(len(segments)-1)
    insets=[];x=0.0
    for k,((lo,hi),span) in enumerate(zip(segments,spans)):
        w=usable*span/sum(spans)
        ins=ax.inset_axes([x,0.0,w,1.0])
        ins.set_xlim(lo,hi)
        style_axis(ins,grid='x')
        ins.spines['top'].set_visible(False);ins.spines['right'].set_visible(False)
        if k>0:
            ins.spines['left'].set_visible(False);ins.tick_params(axis='y',left=False,labelleft=False)
        insets.append(ins);x+=w+gap_frac
    # Conventional paired slashes mark every break on the bottom spine.
    mark=dict(marker=[(-1,-.5),(1,.5)],markersize=6,linestyle='none',
              color=COLORS['edge'],mec=COLORS['edge'],mew=.8,clip_on=False,zorder=6)
    for left,right in zip(insets[:-1],insets[1:]):
        left.plot([1],[0],transform=left.transAxes,**mark)
        right.plot([0],[0],transform=right.transAxes,**mark)
    return insets


def build_s19():
    from analyze_prospective_learning_results import bootstrap_ci
    frame = read('trained_subtree_address','seed_outcomes.csv')
    names=['neuron_shared_k1','correct_subtree_k2','within_neuron_deranged_k2','random_dense_rank2','exact_transport','gated_point_emulation']
    labels=['neuron-shared','correct ancestry','route derangement','random rank-2','exact','gated point']
    colors=[COLORS[k] for k in ('low_rank','shunting','highlight','additive','bp','point_mlp')]
    # Historical bootstrap seed of every condition, keyed by name so the
    # replayed intervals do not depend on the row order drawn below.
    seed_of={name:42000+i for i,name in enumerate(names)}
    cv=NativeCanvas(250/72,1,hgutter_pt=32,margins=Margins(left=95,right=17,top=27,bottom=38))
    a=cv.panel('A',0,0,6,title='Two-branch learning',grid='none')
    b=cv.panel('B',0,6,6,title='Context forgetting',grid='x')
    # A: exact_transport, correct_subtree_k2 and gated_point_emulation give
    # bitwise-identical accuracy in all ten seeds (also backpropagation), so
    # one row carries the three; the identical group leads, the live contrast
    # (neuron-shared) follows, and the two failure controls close the panel.
    rows_a=[('exact_transport','exact = correct ancestry\n= gated point',COLORS['bp']),
            ('neuron_shared_k1','neuron-shared',COLORS['low_rank']),
            ('random_dense_rank2','random rank-2',COLORS['additive']),
            ('within_neuron_deranged_k2','route derangement',COLORS['highlight'])]
    for name in ('correct_subtree_k2','gated_point_emulation','backpropagation'):
        assert np.array_equal(np.sort(frame[frame.condition.eq(name)].test_accuracy.to_numpy(float)),
                              np.sort(frame[frame.condition.eq('exact_transport')].test_accuracy.to_numpy(float)))
    # Axis break: no seed lies in 0.21-0.48 or 0.56-0.72, so those spans are
    # removed and every segment keeps one common scale.
    segs=_broken_x_segments(a,[(0.165,0.225),(0.465,0.585),(0.725,0.845)])
    STRIP=-.32
    def seg_for(x):
        return next(s for s in segs if s.get_xlim()[0]<=x<=s.get_xlim()[1])
    for index,(name,label,color) in enumerate(rows_a):
        values=frame[frame.condition.eq(name)].test_accuracy.to_numpy(float)
        assert len(values)==10
        ordered=np.sort(values)
        ys=index+STRIP+np.linspace(-.08,.08,len(values))
        for s in segs:
            s.scatter(ordered,ys,s=SEED_MS**2,color=color,alpha=SEED_ALPHA,edgecolors='none',zorder=3)
        if name=='random_dense_rank2':
            # Bimodal across seeds (7 collapsed at chance, 3 recovered): a
            # mean and its interval would sit in the empty gap between the
            # modes, so the row shows the median and the two modal counts.
            med=float(np.median(values))
            lowmode=int((values<0.6).sum());highmode=len(values)-lowmode
            assert (lowmode,highmode)==(7,3)
            seg_for(med).plot([med],[index],marker='|',ms=MARKER_MS+4.5,mew=LW_DATA,color=color,ls='none',zorder=5)
            seg_for(med).text(med+.013,index,f'{lowmode} at chance',ha='left',va='center',fontsize=PT_SMALL,color=label_color(color))
            segs[2].text(.727,index,f'{highmode} recovered',ha='left',va='center',fontsize=PT_SMALL,color=label_color(color))
        else:
            m,lo,hi=bootstrap_ci(values,seed=seed_of[name])
            seg_for(m).errorbar(m,index,xerr=[[m-lo],[hi-m]],color=color,marker='D',markerfacecolor='white',
                        ms=MARKER_MS+1.2,lw=LW_ERR,elinewidth=LW_ERR,capsize=ERR_CAPSIZE,zorder=5)
    # Balanced binary task: chance is 0.5.
    segs[1].axvline(.5,color=COLORS['mute'],ls='--',lw=LW_REF,zorder=1)
    segs[1].text(.507,-.52,'chance',ha='left',va='center',fontsize=PT_SMALL,color=COLORS['mute'])
    for s in segs:
        s.set_ylim(len(rows_a)-.45,-.80)
    segs[0].set_yticks(range(len(rows_a)),[r[1] for r in rows_a])
    segs[0].set_xticks([.2]);segs[1].set_xticks([.5]);segs[2].set_xticks([.75,.8])
    segs[1].set_xlabel('Held-out accuracy')
    for index,(name,color) in enumerate(zip(names[:4],colors[:4])):
        values=frame[frame.condition.eq(name)].context_switch_forgetting.to_numpy(float)
        assert len(values)==10
        ordered=np.sort(values)
        b.scatter(ordered,index-.40+np.linspace(-.08,.08,len(values)),s=SEED_MS**2,color=color,alpha=SEED_ALPHA,edgecolors='none',zorder=3)
        m,lo,hi=bootstrap_ci(values,seed=43000+index)
        b.errorbar(m,index,xerr=[[m-lo],[hi-m]],color=color,marker='D',markerfacecolor='white',
                    ms=MARKER_MS+1.2,lw=LW_ERR,elinewidth=LW_ERR,capsize=ERR_CAPSIZE,zorder=5)
    b.set_yticks(range(4),labels[:4]);b.set_ylim(4-.45,-.80)
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
    cv=NativeCanvas(210/72,1,hgutter_pt=30,margins=Margins(left=48,right=17,top=25,bottom=30))
    a=cv.panel('A',0,0,6,title='Route support')
    b=cv.panel('B',0,6,6,title='Direct-type capture, K = 8 (8 cells)',grid='y')
    groups=[g.credit_domain_fraction.to_numpy(float) for _,g in cell_depth.groupby('depth_bin',observed=True)]
    for i,arr in enumerate(groups):
        a.scatter(i+jitter(arr.size,210+i,.07),arr,s=SEED_MS**2,color=COLORS['shunting'],alpha=SEED_ALPHA,edgecolor='white',linewidth=.2)
        errorbar_mean(a,i,arr,COLORS['shunting'],seed=220+i)
    a.set_yticks([0,.1,.2]);a.set_xticks(range(5),['0–2','3–4','5–6','7–9','10+']);a.set_xlabel('topological depth');a.set_ylabel('domain fraction')
    # Squared-energy capture at the eight-channel budget, every cell drawn;
    # the dense PCA oracle leads as the ceiling the other sheets key it as,
    # and the ticks use the shared key's arm names.
    typed=read('figure3','typed_only_compression_curves.csv')
    typed=typed[typed.channels.eq(8)].copy();typed['credit_capture']=1-typed.residual.astype(float)**2
    arms=[('dense PCA oracle','dense PCA\noracle',360),('morphology-aware paths','ancestry\nroutes',370),
          ('random paths','random\npaths',371),('depth-only bins','depth-only\nbins',372),('shuffled ancestry','shuffled\nancestry',373)]
    for i,(method,label,seed) in enumerate(arms):
        arr=typed[typed.method.eq(method)].groupby('root_id').credit_capture.mean().to_numpy(float)
        assert len(arr)==8
        b.scatter(i+jitter(arr.size,seed,.05),arr,s=SEED_MS**2,color=METHOD_COLORS[method],alpha=.60,edgecolor='white',linewidth=.2)
        errorbar_mean(b,i,arr,METHOD_COLORS[method],seed=seed+10)
    b.set_xticks(range(5),[label for _,label,_ in arms]);b.set_xlim(-.6,4.6);b.set_ylim(0,1.0);b.set_yticks([0,.2,.4,.6,.8,1.0]);b.set_ylabel('field capture')
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
