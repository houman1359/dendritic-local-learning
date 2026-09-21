#!/usr/bin/env python3
"""Research visualization of completed exact-target tree constructions.

Reads the corrected, separately versioned DP outputs. Additional depth scores
are deterministic Fourier calculations, not new training or fitted outcomes.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model import tasks,matricize
from constructive import prefix_tasks

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'analysis/morphology_investigation_20260905/structure/constructive_dp_v2'
OUT=ROOT/'analysis/morphology_investigation_20260905/design'
COLORS={'quadratic_matching':'#278365','quartic_partition':'#8055AD','nested_prefix_control':'#C27B26'}
LABELS={'quadratic_matching':'Quadratic matching','quartic_partition':'Quartic partition','nested_prefix_control':'Nested-prefix control'}

def best_bound_at_depth(coeff,max_depth):
 costs={255:0.}
 for mask in range(1,255):
  subset=tuple(i for i in range(8) if mask&(1<<i))
  s=np.linalg.svd(matricize(coeff,subset)[1:],compute_uv=False)
  costs[mask]=round(float(np.sum(s[1:]**2)/(coeff@coeff)),12)
 @lru_cache(None)
 def solve(mask,remaining):
  if mask.bit_count()==1:return 0.
  if remaining==0 or mask.bit_count()>2**remaining:return float('inf')
  first=mask&-mask;sub=(mask-1)&mask;best=float('inf')
  while sub:
   other=mask^sub
   if other and sub&first:
    best=min(best,max(costs[mask],solve(sub,remaining-1),solve(other,remaining-1)))
   sub=(sub-1)&mask
  return best
 return solve(255,max_depth)

def draw_tree(ax,manifest,color):
 children={int(k):v for k,v in manifest['children'].items()}
 root=max(children);positions={};order=[]
 def visit(node,depth):
  if node<8:
   positions[node]=(len(order),4-depth);order.append(node)
  else:
   for child in children[node]:visit(child,depth+1)
   positions[node]=(sum(positions[c][0] for c in children[node])/2,4-depth)
 visit(root,0)
 for node,pair in children.items():
  for child in pair:
   ax.plot([positions[node][0],positions[child][0]],[positions[node][1],positions[child][1]],color='#737F86',lw=1.5,zorder=1)
 for node,(x,y) in positions.items():
  if node<8:
   ax.scatter([x],[y],s=65,marker='s',facecolor='white',edgecolor=color,lw=1.2,zorder=3)
   ax.text(x,y-.33,rf'$x_{{{node+1}}}$',ha='center',va='top',fontsize=10)
  else:
   ax.scatter([x],[y],s=110 if node==root else 80,facecolor=color,edgecolor='white',lw=.8,zorder=3)
 ax.text(positions[root][0],4.48,'root readout',ha='center',fontsize=9,color=color)
 ax.set_xlim(-.65,7.65);ax.set_ylim(-.8,5.4);ax.axis('off')

def main():
 end=pd.read_csv(SOURCE/'adaptive_constructions.csv')
 manifest={z['task_id']:z for z in json.loads((SOURCE/'constructed_trees.json').read_text())}
 tasklist=tasks()+prefix_tasks()
 assert len(end)==164 and end.normalized_mse.max()<1e-20
 rows=[]
 for task in tasklist:
  for depth in [3,4,5]:
   rows.append(dict(task_id=task['task_id'],family=task['family'],maximum_depth=depth,
    best_centered_cut_bound=best_bound_at_depth(task['coefficients'],depth)))
 scores=pd.DataFrame(rows)
 OUT.mkdir(parents=True,exist_ok=True)
 scores.to_csv(OUT/'constructive_depth_certificate.csv',index=False)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(11.2,7.8))
 gs=fig.add_gridspec(2,2,left=.075,right=.97,top=.89,bottom=.145,wspace=.30,hspace=.52,height_ratios=[1.15,1])
 for index,tid in enumerate(['matching_000','nested_prefix_000']):
  ax=fig.add_subplot(gs[0,index]);fam='quadratic_matching' if index==0 else 'nested_prefix_control'
  draw_tree(ax,manifest[tid],COLORS[fam])
  ax.set_title(('A    Parallel interactions: minimum depth 3' if index==0 else 'B    Nested interactions: minimum depth 4'),loc='left',fontsize=11,fontweight='bold',pad=22)
  if index==0:
   formula=r'$f(x)=\frac{1}{2}(x_1x_2+x_3x_4+x_5x_6+x_7x_8)$'
  else:
   permutation=prefix_tasks()[0]['permutation']
   formula=r'$f(z)=\frac{1}{2}\sum_{k\in\{2,4,6,8\}}\,\prod_{j=1}^{k}z_j$'
   assignment='$z=('+','.join(f'x_{{{int(i)+1}}}' for i in permutation)+')$'
   ax.text(.5,-.07,assignment,ha='center',transform=ax.transAxes,fontsize=9,color='#606970')
  ax.text(.5,1.005,formula,ha='center',transform=ax.transAxes,fontsize=11)
 c=fig.add_subplot(gs[1,0])
 c.set_title('C    Exact depth constraint, across all labeled trees',loc='left',fontsize=10.5,pad=14)
 for family,color in COLORS.items():
  z=scores[scores.family.eq(family)].groupby('maximum_depth').best_centered_cut_bound
  lower,upper=z.min(),z.max()
  assert np.allclose(lower,upper,atol=1e-12)
  c.plot(lower.index,lower.values,color=color,marker={'quadratic_matching':'o','quartic_partition':'s','nested_prefix_control':'^'}[family],ms=6,lw=1.4,label=LABELS[family],zorder=4 if family=='quadratic_matching' else 3)
 c.set_xticks([3,4,5]);c.set_xlabel('Maximum root-to-leaf edge depth');c.set_ylabel('Minimum centered-cut bound (NMSE)')
 c.set_ylim(-.025,max(scores.best_centered_cut_bound)*1.18+.01)
 c.spines[['top','right']].set_visible(False);c.grid(axis='y',color='#E3E7EB',lw=.6);c.set_axisbelow(True)
 c.legend(frameon=False,fontsize=8,loc='upper right')
 c.text(.42,.38,'Zero bound ⇔ exact representation\nfor this multi-affine tree class',transform=c.transAxes,fontsize=8,color='#606970')
 d=fig.add_subplot(gs[1,1]);d.axis('off')
 d.set_title('D    Constructive verification; fixed resources',loc='left',fontsize=10.5,pad=14)
 d.text(0,.91,'All trees: 28 coefficients · 14 edges · 8 named inputs',fontsize=9)
 d.text(0,.76,'Family',fontweight='bold',fontsize=9);d.text(.59,.76,'Tasks',ha='center',fontweight='bold',fontsize=9);d.text(.83,.76,'Exact depth',ha='center',fontweight='bold',fontsize=9)
 for i,(family,color) in enumerate(COLORS.items()):
  z=end[end.family.eq(family)];y=.64-i*.12
  d.text(0,y,LABELS[family],color=color,fontsize=9)
  d.text(.59,y,str(len(z)),ha='center',fontsize=9)
  d.text(.83,y,str(int(z.depth.iloc[0])),ha='center',fontsize=9)
 d.text(0,.20,f'Maximum NMSE: {end.normalized_mse.max():.2g}.\nMaximum |coefficient|: 1 + numerical rounding.',fontsize=8.3,linespacing=1.5)
 d.text(0,.015,'Primary families: sensitivity spectrum = (¼, …, ¼).\nNested control: (¼, ¼, ½, ½, ¾, ¾, 1, 1); rank remains 8.',fontsize=8.3,color='#606970',linespacing=1.5)
 fig.text(.5,.975,'Interaction structure determines compatible input grouping and required depth',ha='center',fontsize=13)
 fig.text(.5,.033,'Exploratory full-target oracle construction: exact Fourier tensor from all 256 inputs; dynamic programming over input subsets.\nScope: nodes affine in each child, u = a + bℓ + cr + dℓr. No inference about general nonlinear or conductance trees.',ha='center',fontsize=8,color='#606970',linespacing=1.45)
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'morphology_constructive_diagnostic.{ext}',dpi=190)
 plt.close(fig)
 sources={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ['protocol.json','adaptive_constructions.csv','constructed_trees.json']}
 (OUT/'constructive_figure_provenance.json').write_text(json.dumps(dict(builder='scripts/morphology_structure/plot_constructive.py',sources=sources,depth_certificate='Exact minimax dynamic programming constrained by maximum depth, performed for every included target; all outcomes retained',example_rule='First enumerated matching and first seeded nested-prefix target; no selection by fitted outcome',scope='Corrected constructive_dp_v2 data; full-target oracle, specific multi-affine class'),indent=2)+'\n')
 print(scores.groupby(['family','maximum_depth']).best_centered_cut_bound.agg(['min','max']).to_string())

if __name__=='__main__':main()
