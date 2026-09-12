#!/usr/bin/env python3
"""Focal-control detail from frozen cell and site summaries, with bin counts.

2026-09-11 (S29 visual review): A orders its columns shunt / current
injection / reassigned and pairs only the two matched conditions; C labels
every sensitivity row with both held parameters, separates the scale and
reversal families, draws the reference interval as a band, keeps the whole
interval inside the axis and prints the cells-positive count per row; D is a
paired plot of the same eight cells with a two-line axis label.  Only
``source_data/`` tables are read; the depth-interval export is deterministic.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import (COLORS,ERR_CAPSIZE,LW_DATA,LW_ERR,LW_HAIR,LW_REF,MARKER_MS,PT_BASE,PT_SMALL,
                           SEED_ALPHA,SEED_MS,Margins,NativeCanvas,tint_patch)
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source_data/figure4'
CONTRAST_LABEL='shunt − current-injection\nlocalization'
# one declared reserve on every panel: the column lock then gives A and B
# (and every other same-span pair) one axes width by construction
DECLARED=dict(left=12.0,right=14.0)

def interval(values,seed):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    lo,hi=np.quantile(rng.choice(values,(10000,len(values)),replace=True).mean(axis=1),[.025,.975])
    return values.mean(),lo,hi

def paired(ax,frames,colors,labels,*,seeds,pair=None,xpos=None):
    """Per-cell points with a light pairing line and a mean +/- 95 % CI diamond."""
    xpos=list(range(len(frames))) if xpos is None else list(xpos)
    pair=list(range(len(frames))) if pair is None else list(pair)
    cells=np.column_stack([np.asarray(f,float) for f in frames])
    for row in cells:
        ax.plot([xpos[i] for i in pair],[row[i] for i in pair],color=COLORS['mute'],lw=LW_HAIR,alpha=.4,zorder=1)
    for i,(vals,color,seed) in enumerate(zip(cells.T,colors,seeds)):
        mean,lo,hi=interval(vals,seed)
        ax.scatter(xpos[i]+np.linspace(-.06,.06,len(vals)),vals,s=12,color=color,alpha=SEED_ALPHA+.05,zorder=2,edgecolors='none')
        ax.errorbar(xpos[i],mean,yerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.8,color=color,capsize=ERR_CAPSIZE,lw=LW_ERR,zorder=3)
    ax.set_xticks(xpos,labels);ax.axhline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
    return cells

def build():
    primary=pd.read_csv(SOURCE/'cell_primary_contrasts.csv')
    direct=pd.read_csv(SOURCE/'direct_typed_cell_primary_contrasts.csv')
    focal=pd.read_csv(SOURCE/'focal_localization.csv')
    cv=NativeCanvas(390/72,2,hgutter_pt=35,vgutter_pt=67,margins=Margins(left=50,right=8,top=20,bottom=42))
    a=cv.panel('A',0,0,6,title='Within-cell controls',grid='y')
    b=cv.panel('B',0,6,6,title='Focal depth: varying cell coverage',grid='y')
    c=cv.panel('C',1,0,7,title='Synaptic scales and reversal',grid='x')
    d=cv.panel('D',1,7,5,title='Direct presynaptic types',grid='y')
    for name in 'ABCD':cv.declare_reserve(name,**DECLARED)
    # the supplement paste cuts each panel at its letter's x - 3 pt, so D's
    # two-line y label must stay right of the letter D
    cv.declare_reserve('D',left=27.0)
    # A: the two matched conditions are paired cell by cell; the reassignment
    # control is a different comparison, so it stands apart and unpaired.
    cols=['focal_shunt_localization','matched_additive_localization','shunt_depth_shuffled_localization']
    paired(a,[primary[k] for k in cols],[COLORS['shunting'],COLORS['additive'],COLORS['point_mlp']],
           ['shunt','current\ninjection','reassigned'],seeds=(442,440,441),pair=[0,1],xpos=[0,1,2.35])
    a.set_xlim(-.5,2.85);a.set_ylabel('localization index')
    percell=focal[focal.perturbation.eq('focal shunt')].groupby(['root_id','focal_topological_depth']).localization_index.mean().reset_index()
    bins=[];means=[];lows=[];highs=[];counts=[];interval_rows=[]
    for depth,g in percell.groupby('focal_topological_depth'):
        vals=g.localization_index.to_numpy();mean,lo,hi=interval(vals,21000+int(depth))
        bins.append(depth);means.append(mean);lows.append(lo);highs.append(hi);counts.append(len(vals))
        interval_rows.append(dict(depth=int(depth),n_cells=len(vals),mean=mean,ci95_low=lo,ci95_high=hi,bootstrap_draws=10000))
        b.scatter(np.repeat(depth,len(vals)),vals,s=8,color=COLORS['shunting'],alpha=.30)
    export=ROOT/"source_data/review_focal_depth";export.mkdir(exist_ok=True)
    pd.DataFrame(interval_rows).to_csv(export/"figure_S21_depth_intervals.csv",index=False)
    means=np.array(means)
    b.errorbar(bins,means,yerr=[means-np.array(lows),np.array(highs)-means],fmt='o',ms=3,color=COLORS['shunting'],capsize=2,lw=LW_DATA)
    b.set_xticks(bins,[f'{x}\n{n}' for x,n in zip(bins,counts)]);b.tick_params(axis='x',labelsize=PT_SMALL)
    b.set_xlabel('topological depth / contributing cells');b.set_ylabel('localization index')
    b.axhline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
    # C: every row names both held parameters; the reference row's interval is
    # the band behind all rows, and the two families are separated by a gap.
    specs=[('scale0p1_summary.json','E/I 0.10\nreversal −0.2',0.0),
           ('summary.json','E/I 0.35\nreversal −0.2',1.0),
           ('scale1p0_summary.json','E/I 1.00\nreversal −0.2',2.0),
           ('irevm0p5_summary.json','E/I 0.35\nreversal −0.5',3.45),
           ('irev0_summary.json','E/I 0.35\nreversal 0.0',4.45)]
    rows=[]
    for name,label,y in specs:
        r=json.loads((SOURCE/name).read_text())['primary_contrast']
        rows.append(dict(label=label,y=y,mean=r['mean_shunt_minus_additive'],lo=r['cell_bootstrap_ci95'][0],hi=r['cell_bootstrap_ci95'][1],
                         positive=int(r['cells_positive']),n=int(r['n_cells'])))
    ref=rows[1];assert ref['label'].startswith('E/I 0.35\nreversal −0.2')
    xlo=-0.012;xhi=1.05*max(r['hi'] for r in rows);ytop=-1.35;ybot=rows[-1]['y']+.6
    c.set_xlim(xlo,xhi);c.set_ylim(ybot,ytop);c.set_yticks([])
    for spine in ('left','top','right'):c.spines[spine].set_visible(False)
    tint_patch(c,('rect',ref['lo'],ytop,ref['hi']-ref['lo'],ybot-ytop),color=COLORS['shunting'],pct=14,edge=False,radius_pt=0.0,zorder=0.3,clip_on=True)
    c.annotate('reference interval',xy=(ref['lo'],-0.85),xycoords='data',xytext=(-2.5,0.0),textcoords='offset points',ha='right',va='center',fontsize=PT_BASE,color=COLORS['mute'])
    c.axvline(0,color=COLORS['mute'],ls='--',lw=LW_REF,zorder=1)
    for r in rows:
        y=r['y']
        c.plot([xlo,xlo],[y-.3,y+.3],color=COLORS['edge'],lw=LW_HAIR,clip_on=False,zorder=1.5,solid_capstyle='butt')
        c.errorbar(r['mean'],y,xerr=[[r['mean']-r['lo']],[r['hi']-r['mean']]],fmt='o',ms=3.8,color=COLORS['shunting'],capsize=ERR_CAPSIZE,lw=LW_ERR,zorder=3,
                   markerfacecolor='white' if r is ref else COLORS['shunting'],markeredgewidth=LW_ERR)
        c.annotate(r['label'],xy=(0.0,y),xycoords=('axes fraction','data'),xytext=(-4.0,0.0),textcoords='offset points',
                   ha='right',va='center',fontsize=PT_BASE,color=COLORS['ink'],linespacing=1.15,annotation_clip=False)
        c.annotate(f"{r['positive']}/{r['n']} cells",xy=(1.0,y),xycoords=('axes fraction','data'),xytext=(3.0,0.0),textcoords='offset points',
                   ha='left',va='center',fontsize=PT_BASE,color=COLORS['mute'],annotation_clip=False)
    cv.declare_reserve('C',right=36.0)
    c.set_xticks([0,.04,.08,.12]);c.set_xlabel(CONTRAST_LABEL.replace('\n',' '));c.tick_params(axis='y',length=0)
    # D: the same eight cells, paired across the two contact sets
    assert list(primary.root_id)==list(direct.root_id)
    cells=paired(d,[primary.shunt_minus_additive,direct.shunt_minus_additive],[COLORS['shunting'],COLORS['pathway']],
                 ['all mapped','direct typed'],seeds=(510,511))
    assert int((cells[:,1]>cells[:,0]).sum())==8
    d.set_xlim(-.45,1.45);d.set_ylabel(CONTRAST_LABEL)
    return cv.save(ROOT/'figures/supplementary/figure_S21_panels_A-D.pdf',name='figure_S21_panels_A-D')
if __name__=='__main__':build()
