#!/usr/bin/env python3
"""Focal-control detail from frozen cell and site summaries, with bin counts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import COLORS,LW_DATA,LW_HAIR,LW_REF,PT_SMALL,Margins,NativeCanvas
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source_data/figure4'

def interval(values,seed):
    values=np.asarray(values,float);rng=np.random.default_rng(seed)
    lo,hi=np.quantile(rng.choice(values,(10000,len(values)),replace=True).mean(axis=1),[.025,.975])
    return values.mean(),lo,hi

def build():
    primary=pd.read_csv(SOURCE/'cell_primary_contrasts.csv')
    direct=pd.read_csv(SOURCE/'direct_typed_cell_primary_contrasts.csv')
    focal=pd.read_csv(SOURCE/'focal_localization.csv')
    cv=NativeCanvas(390/72,2,hgutter_pt=35,vgutter_pt=67,margins=Margins(left=45,right=13,top=20,bottom=42))
    a=cv.panel('A',0,0,6,title='Within-cell controls',grid='y')
    b=cv.panel('B',0,6,6,title='Focal depth: varying cell coverage',grid='y')
    c=cv.panel('C',1,0,6,title='Synaptic scales and reversal',grid='x')
    d=cv.panel('D',1,6,6,title='Direct presynaptic types',grid='y')
    cols=['matched_additive_localization','shunt_depth_shuffled_localization','focal_shunt_localization']
    for _,row in primary.iterrows():a.plot(range(3),row[cols],color=COLORS['mute'],lw=LW_HAIR,alpha=.4)
    for i,(col,color) in enumerate(zip(cols,[COLORS['additive'],COLORS['mute'],COLORS['shunting']])):
        vals=primary[col].to_numpy();mean,lo,hi=interval(vals,440+i)
        a.scatter(i+np.linspace(-.045,.045,len(vals)),vals,s=12,color=color,alpha=.65)
        a.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.8,color=color,capsize=2,lw=LW_DATA)
    a.set_xticks(range(3),['current\ninjection','reassigned','shunt']);a.set_ylabel('localization index');a.axhline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
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
    specs=[('scale0p1_summary.json','E/I scale 0.10'),('summary.json','E/I scale 0.35'),('scale1p0_summary.json','E/I scale 1.00'),('irevm0p5_summary.json','reversal −0.5'),('irev0_summary.json','reversal 0.0')]
    for i,(name,label) in enumerate(specs):
        r=json.loads((SOURCE/name).read_text())['primary_contrast'];mean=r['mean_shunt_minus_additive'];lo,hi=r['cell_bootstrap_ci95']
        c.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',ms=3.8,color=COLORS['shunting'],capsize=2,lw=LW_DATA)
    c.set_yticks(range(5),[s[1] for s in specs]);c.tick_params(axis='y',labelsize=PT_SMALL);c.invert_yaxis()
    c.set_xlabel('localization: shunt − current injection');c.axvline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
    for i,(frame,color) in enumerate([(primary,COLORS['shunting']),(direct,COLORS['pathway'])]):
        vals=frame.shunt_minus_additive.to_numpy();mean,lo,hi=interval(vals,510+i)
        d.scatter(i+np.linspace(-.045,.045,len(vals)),vals,s=12,color=color,alpha=.7)
        d.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.8,color=color,capsize=2,lw=LW_DATA)
    d.set_xticks([0,1],['all mapped\n8 cells','direct typed\n8 cells']);d.set_ylabel('shunt − current-injection localization')
    d.axhline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
    return cv.save(ROOT/'figures/supplementary/figure_S21_panels_A-D.pdf',name='figure_S21_panels_A-D')
if __name__=='__main__':build()
