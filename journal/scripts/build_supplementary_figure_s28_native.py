#!/usr/bin/env python3
"""Explicit pairwise credit contrasts; no additive causal decomposition."""
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import COLORS,LW_DATA,LW_REF,PT_SMALL,Margins,NativeCanvas
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source_data/point_dendrite_credit_controls'
SPECS=[('full_bp','full BP','bp','o','-'),('soma_broadcast_bp','soma broadcast / BP optimizer','additive','s','--'),('soma_broadcast_matched','soma broadcast / LocalCA optimizer','highlight','v','--'),('local_path','exact-path LocalCA','pathway','^','-.'),('local_shared','shared-soma LocalCA','local','D',':')]
CONTRASTS=[
 ('full_bp_minus_soma_broadcast','full_bp','soma_broadcast_bp','BP − soma broadcast (BP optimizer)'),
 ('bp_optimizer_broadcast_minus_matched_broadcast','soma_broadcast_bp','soma_broadcast_matched','Soma broadcast: BP optimizer − LocalCA optimizer'),
 ('matched_broadcast_minus_local_shared','soma_broadcast_matched','local_shared','Soma-broadcast autograd − shared-soma LocalCA (LocalCA optimizer)'),
 ('local_path_minus_local_shared','local_path','local_shared','Exact-path − shared-soma LocalCA (LocalCA optimizer)'),
 ('full_bp_minus_local_path','full_bp','local_path','BP − exact-path LocalCA (respective optimizers)')]

def build():
 summary=pd.read_csv(SOURCE/'condition_summary.csv');contrasts=pd.read_csv(SOURCE/'paired_contrasts.csv').set_index('contrast')
 seeds=pd.read_csv(SOURCE/'combined_seed_outcomes.csv')
 seeds=seeds[seeds.architecture.eq('serial_tree')&seeds.regime.eq('aligned')&seeds.depth.eq(3)]
 wide=seeds.pivot(index='seed',columns='credit',values='test_accuracy')
 cv=NativeCanvas(455/72,2,row_weights=[150,190],hgutter_pt=30,vgutter_pt=50,margins=Margins(left=46,right=15,top=20,bottom=34))
 a=cv.panel('A',0,0,12,title='Credit-coordinate ladder across physical depth',grid='y')
 b=cv.panel('B',1,0,12,title='Explicit paired D3 comparisons',grid='x')
 for credit,label,color,marker,ls in SPECS:
  f=summary[summary.architecture.eq('serial_tree')&summary.regime.eq('aligned')&summary.credit.eq(credit)].sort_values('depth')
  mean=f.mean_test_accuracy.to_numpy();color=COLORS[color]
  a.errorbar(f.depth,mean,yerr=[mean-f.ci95_low_test_accuracy,f.ci95_high_test_accuracy-mean],color=color,marker=marker,ls=ls,lw=LW_DATA,ms=3.5,capsize=2,label=label)
 a.set_xticks([1,2,3],['D1','D2','D3']);a.set_ylim(.44,1.06);a.set_xlim(.75,3.25);a.set_xlabel('serial physical depth');a.set_ylabel('held-out accuracy')
 a.legend(loc='upper left',frameon=False,fontsize=PT_SMALL,ncol=2,columnspacing=1.7,handlelength=2.2)
 for i,(key,left,right,label) in enumerate(CONTRASTS):
  r=contrasts.loc[key+'__aligned__d3'];mean=100*r.mean_difference;lo=100*r.ci95_low;hi=100*r.ci95_high
  vals=100*(wide[left]-wide[right]).dropna().to_numpy();assert len(vals)==10
  b.scatter(vals,i+np.linspace(-.07,.07,len(vals)),s=11,color=COLORS['additive'],alpha=.42)
  b.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='D',ms=3.9,color=COLORS['additive'],lw=LW_DATA,capsize=2)
  b.text(-1.0,i-.29,label,fontsize=PT_SMALL,color=COLORS['ink'],va='bottom')
 b.set_ylim(4.25,-.60);b.set_xlim(-1.4,18);b.set_yticks([]);b.set_xlabel('paired accuracy difference (percentage points)')
 b.axvline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
 return cv.save(ROOT/'figures/supplementary/figure_S28_panels_A-B.pdf',name='figure_S28_panels_A-B')
if __name__=='__main__':build()
