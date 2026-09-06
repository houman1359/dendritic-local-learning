#!/usr/bin/env python3
"""Four-panel research figure from completed exhaustive morphology outcomes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'analysis/morphology_investigation_20260905/structure'
OUT=ROOT/'analysis/morphology_investigation_20260905/design'
COLORS={'quadratic_matching':'#278365','quartic_partition':'#8055AD'}
LABELS={'quadratic_matching':'Quadratic matchings (105)','quartic_partition':'Quartic partitions (35)'}

def panel(ax,letter,title):
 ax.set_title(title,fontsize=10.5,pad=12,loc='left')
 ax.text(-.09,1.10,letter,transform=ax.transAxes,fontsize=13,fontweight='bold',va='bottom')

def clean(ax,grid='y'):
 for k in ['top','right']:ax.spines[k].set_visible(False)
 for k in ['bottom','left']:ax.spines[k].set_color('#889096')
 ax.tick_params(color='#889096',labelsize=8)
 if grid:ax.grid(axis=grid,color='#E3E7EB',linewidth=.55,zorder=0)
 ax.set_axisbelow(True)

def main():
 end=pd.read_csv(SOURCE/'candidate_outcomes.csv')
 policy=pd.read_csv(SOURCE/'audit_policy_summary.csv')
 assert len(end)==1680 and set(end.family)==set(COLORS)
 OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7,'lines.linewidth':1.3})
 fig=plt.figure(figsize=(11.2,8.0))
 gs=fig.add_gridspec(2,2,left=.075,right=.98,top=.895,bottom=.12,wspace=.40,hspace=.43,height_ratios=[1,1.08])
 a=fig.add_subplot(gs[0,0]);a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
 panel(a,'A','Same sensitivity spectrum; different interactions')
 a.text(0,.91,'All perfect matchings',color=COLORS['quadratic_matching'],fontsize=10)
 a.text(0,.79,r'$f_M(x)=\frac{1}{2}\sum_{(i,j)\in M}x_i x_j$',fontsize=15)
 a.text(0,.61,'All unordered four-plus-four partitions',color=COLORS['quartic_partition'],fontsize=10)
 a.text(0,.49,r'$f_A(x)=\frac{1}{2}\,[\prod_{i\in A}x_i+\prod_{i\notin A}x_i]$',fontsize=14)
 a.text(0,.31,r'Both: $\mathbb{E}[\nabla_x f\nabla_x f^{\mathsf{T}}]=I_8/4$; effective rank $=8$',fontsize=10)
 a.add_patch(Rectangle((0,.015),.98,.205,facecolor='#F3F5F7',edgecolor='#D5DBE0',lw=.7))
 a.text(.02,.16,r'Scalar tree: $u=a+b\ell+cr+d\ell r$',fontsize=10)
 a.text(.02,.09,'8 inputs · 7 internal units · 28 coefficients · 14 edges',fontsize=8.6)
 a.text(.02,.035,'One input per leaf; root readout only; no dense compensation',fontsize=8.1)
 # B contains two matched axes, one for each valid lower bound.
 bg=gs[0,1].subgridspec(1,2,wspace=.23)
 bx=[]
 for j,(metric,title) in enumerate([('full_cut_bound','Full matrix: rank ≤ 2'),('centered_cut_bound','Centered rows: rank ≤ 1')]):
  ax=fig.add_subplot(bg[0,j]);bx.append(ax);clean(ax,grid=None)
  ax.plot([0,.82],[0,.82],ls='--',lw=.8,color='#9A9FA5')
  for family,color in COLORS.items():
   z=end[end.family.eq(family)].copy();z['xx']=z[metric].round(7);z['yy']=z.normalized_mse.round(7)
   counts=z.groupby(['xx','yy']).size().reset_index(name='count')
   ax.scatter(counts.xx,counts.yy,s=18+7*np.sqrt(counts['count']),marker='o' if family=='quadratic_matching' else '^',facecolors=color if family=='quadratic_matching' else 'white',edgecolors=color,alpha=.65,linewidths=.9,zorder=3)
  ax.set_xlim(-.045,.82);ax.set_ylim(-.045,.82);ax.set_xticks([0,.25,.5,.75]);ax.set_yticks([0,.25,.5,.75]);ax.set_title(title,fontsize=9,pad=8)
  ax.set_xlabel('MSE lower bound / variance',fontsize=8)
  if j==0:ax.set_ylabel('Best fitted population NMSE')
  else:ax.tick_params(labelleft=False)
  ax.set_aspect('equal',adjustable='box')
 box=gs[0,1].get_position(fig)
 fig.text(box.x0-.035,box.y1+.022,'B',fontsize=13,fontweight='bold')
 fig.text(box.x0,box.y1+.025,'The centered constraint is more informative',fontsize=10.5)
 fig.text(box.x0,box.y0-.035,'Marker area reflects coincident candidate–task cases; dashed line is equality.',fontsize=7.5,color='#606970')
 c=fig.add_subplot(gs[1,0]);panel(c,'C','Shape effects at matched leaf assignments');clean(c)
 shapes=['balanced','mixed','comb'];xx=np.arange(3)
 for j,(family,color) in enumerate(COLORS.items()):
  z=end[end.family.eq(family)].groupby('shape').agg(error=('normalized_mse','mean'),bound=('centered_cut_bound','mean')).loc[shapes]
  offset=(-.18 if j==0 else .18)
  c.bar(xx+offset,z.error,width=.32,color=color,alpha=.78,label=LABELS[family],zorder=2)
  c.scatter(xx+offset,z.bound,marker='D',s=24,facecolor='white',edgecolor='#34434B',lw=.9,zorder=4)
  for x,y in zip(xx+offset,z.error):c.text(x,y+.025,f'{y:.3f}',ha='center',fontsize=8,color=color)
 c.set_xticks(xx,['balanced','3 + 5 split','comb']);c.set_ylim(0,.88);c.set_ylabel('Mean best-restart population NMSE')
 c.text(.02,.97,'Same four input assignments per shape',transform=c.transAxes,va='top',fontsize=8,color='#606970')
 c.legend(handles=[Line2D([],[],marker='D',ls='',mfc='white',mec='#34434B',label='Mean centered lower bound')],loc='lower right',frameon=False,fontsize=8)
 d=fig.add_subplot(gs[1,1]);panel(d,'D','Selection within the fixed candidate pool');clean(d,grid='x')
 keys=['centered_cut_bound','full_cut_bound','centered_cut_sum','two_sweep_pilot','best_fixed_in_hindsight','uniform_random_expectation']
 names=['Centered-cut bound','Full-cut bound','Sum of centered tails\n(heuristic)','Two-sweep fitting pilot','Best fixed in hindsight','Random expectation']
 for j,(family,color) in enumerate(COLORS.items()):
  z=policy[policy.family.eq(family)].set_index('policy').loc[keys]
  yy=np.arange(len(keys))+(-.12 if j==0 else .12)
  d.scatter(z.mean_regret,yy,s=37,marker='o' if j==0 else '^',color=color,edgecolor='white',linewidth=.5,zorder=4)
 d.axvline(0,color='#9A9FA5',lw=.8,ls='--');d.set_yticks(range(len(keys)),names,fontsize=8.3);d.invert_yaxis();d.set_xlim(-.009,.29);d.set_xlabel('Mean regret (excess population NMSE)')
 handles=[Line2D([],[],color=color,marker='o' if fam=='quadratic_matching' else '^',lw=0,label=LABELS[fam]) for fam,color in COLORS.items()]
 fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.53,.995),frameon=False,fontsize=9)
 fig.text(.5,.027,'Exhaustive finite-domain diagnostic: all 256 inputs; 12 candidates; 4 ALS restarts × 32 sweeps.\nBounds are specific to affine-in-child, multi-affine trees; no claim about general conductance dendrites.',ha='center',fontsize=8,color='#606970',linespacing=1.4)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'morphology_structure_diagnostic.{ext}',dpi=190,metadata={'Creator':'morphology_structure/plot_results.py'} if ext=='pdf' else None)
 plt.close(fig)
 (OUT/'figure_provenance.json').write_text(json.dumps({'builder':'scripts/morphology_structure/plot_results.py','sources':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['candidate_outcomes.csv','audit_policy_summary.csv','audit_report.json']},'panel_A':'exact target definitions, input-gradient covariance and declared model constraints','panel_B':'all1680 candidate/task results; each endpoint is best of4 ALS restarts, paired with full/centered bounds; marker sizes aggregate coincident points','panel_C':'family means over all tasks and same4 assignments per shape; diamonds mean valid lower bound','panel_D':'tolerance-corrected policy means within fixed12candidate pool; no inferential CIs on exhaustive families','interpretation':'gap above bound does not distinguish optimization error from bound looseness; input-gradient covariance is not parameter-gradient covariance'},indent=2)+'\n')
if __name__=='__main__':main()
