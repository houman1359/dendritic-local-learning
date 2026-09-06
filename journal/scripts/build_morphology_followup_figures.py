#!/usr/bin/env python3
"""Native vector supplementary figures from copied, hash-traced study tables.

No fitting, prediction selection, or frozen investigation output is modified.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from journal_style import (COLORS,style_direct_color_labels,FIG_W,PT_SMALL,PT_ANNOT,PT_LABEL,PT_TITLE,
                           PANEL_LABEL_PT,LW_DATA,LW_EDGE,LW_REF,apply_neurips_style)

ROOT=Path(__file__).resolve().parents[1]
STRUCT=ROOT/'source_data/morphology_structure'
DYN=ROOT/'source_data/morphology_finite_horizon'
OUT=ROOT/'figures/supplementary'
FC={'quadratic_matching':COLORS['shunting'],'quartic_partition':COLORS['oracle'],'nested_prefix_control':COLORS['local']}
ARMS=['feedback_only','joint_forward_feedback']
ARM_NAMES=['Feedback only','Joint transfer']
METHODS=['original_scalar','development_best','rank_only','pilot16','gaussian_plugin_fullbatch','gaussian_plugin_sgd','observed_count_cheapest','gaussian_oracle_sgd']
NAMES={'original_scalar':'original scalar','development_best':'fixed / max-budget',
 'rank_only':'generating-rank policy','pilot16':'16-step pilot',
 'gaussian_plugin_fullbatch':'Gaussian full-batch','gaussian_plugin_sgd':'Gaussian SGD (primary)',
 'observed_count_cheapest':'observed context count','gaussian_oracle_sgd':'population oracle',
 'empirical_split_fullbatch':'empirical full-batch','candidate_training':'all 256-update fits'}
MC={m:c for m,c in zip(METHODS,[COLORS['mute'],COLORS['point_mlp'],COLORS['point_mlp'],COLORS['local'],'#648C9B',COLORS['shunting'],COLORS['oracle'],'#333B41'])}

def clean(ax,grid=None):
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(labelsize=PT_SMALL,width=LW_EDGE,length=3)
 if grid:ax.grid(axis=grid,color='#E4E8EB',lw=.55);ax.set_axisbelow(True)

def title(ax,letter,text):
 ax.set_title(text,loc='left',fontsize=PT_TITLE,pad=10)
 ax.text(-.10,1.095,letter,transform=ax.transAxes,fontsize=PANEL_LABEL_PT,fontweight='bold',va='bottom')

def save(fig,number,panels,sources):
 style_direct_color_labels(fig, colors=[*FC.values(), *MC.values(), *COLORS.values(), '#648C9B'])
 stem=f'figure_S{number:02d}_panels_A-{panels}'
 fig.savefig(OUT/f'{stem}.pdf');fig.savefig(OUT/f'{stem}.png',dpi=180)
 plt.close(fig)
 import inspect
 builder=Path(inspect.stack()[1].filename).resolve()
 manifest={'figure':f'S{number}','panels':f'A-{panels}',
  'builder':str(builder.relative_to(ROOT)),
  'builder_sha256':hashlib.sha256(builder.read_bytes()).hexdigest(),
  'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
  'scope':'Rendering only; source outcomes and protocol unchanged'}
 (OUT/f'{stem}.provenance.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(stem,flush=True)

def s36():
 end=pd.read_csv(STRUCT/'candidate_outcomes.csv')
 policy=pd.read_csv(STRUCT/'audit_policy_summary.csv')
 fig=plt.figure(figsize=(FIG_W,5.65))
 gs=fig.add_gridspec(2,2,left=.10,right=.97,top=.90,bottom=.115,wspace=.42,hspace=.52)
 a=fig.add_subplot(gs[0,0]);a.axis('off');title(a,'A','Fixed spectrum; different interactions')
 a.text(0,.87,'All 105 perfect matchings',color=FC['quadratic_matching'],fontsize=PT_ANNOT)
 a.text(0,.71,r'$f_M=\frac{1}{2}\sum_{(i,j)\in M}x_i x_j$',fontsize=PT_LABEL)
 a.text(0,.52,'All 35 four-plus-four partitions',color=FC['quartic_partition'],fontsize=PT_ANNOT)
 a.text(0,.36,r'$f_A=\frac{1}{2}(\prod_{i\in A}x_i+\prod_{i\notin A}x_i)$',fontsize=PT_LABEL)
 a.text(0,.16,r'$\mathbb{E}[\nabla_x f\nabla_x f^\mathsf{T}]=I_8/4$; rank $=8$',fontsize=PT_ANNOT)
 a.text(0,-.01,'28 coefficients · 14 edges · root-only output',fontsize=PT_SMALL)
 b=fig.add_subplot(gs[0,1]);b.axis('off');title(b,'B','The centered constraint is stronger')
 for j,(metric,label) in enumerate([('full_cut_bound','full rank ≤ 2'),('centered_cut_bound','centered rank ≤ 1')]):
  ax=b.inset_axes([j*.55,.08,.45,.77]);clean(ax)
  for fam,col in list(FC.items())[:2]:
   z=end[end.family.eq(fam)].copy();z['x']=z[metric].round(7);z['y']=z.normalized_mse.round(7)
   z=z.groupby(['x','y']).size().reset_index(name='n')
   ax.scatter(z.x,z.y,s=8+3*np.sqrt(z.n),marker='o' if fam=='quadratic_matching' else '^',facecolors=col if fam=='quadratic_matching' else 'white',edgecolors=col,alpha=.7,lw=.7,zorder=3)
  ax.plot([0,.8],[0,.8],color='#A1A8AE',ls='--',lw=LW_REF)
  ax.set_xlim(-.05,.82);ax.set_ylim(-.05,.82);ax.set_xticks([0,.4,.8]);ax.set_yticks([0,.4,.8]);ax.set_title(label,fontsize=PT_SMALL,pad=5)
  if j==0:ax.set_ylabel('best fitted NMSE',fontsize=PT_SMALL)
  else:ax.tick_params(labelleft=False)
 b.text(.5,-.10,'cut lower bound / target variance',ha='center',fontsize=PT_SMALL)
 c=fig.add_subplot(gs[1,0]);clean(c,'y');title(c,'C','Shape effects at matched assignments')
 for j,(fam,col) in enumerate(list(FC.items())[:2]):
  z=end[end.family.eq(fam)].groupby('shape')[['normalized_mse','centered_cut_bound']].mean().loc[['balanced','mixed','comb']]
  x=np.arange(3)+(j-.5)*.34
  c.bar(x,z.normalized_mse,width=.3,color=col,alpha=.8)
  c.scatter(x,z.centered_cut_bound,s=20,marker='D',facecolor='white',edgecolor='#45525A',lw=.7,zorder=3)
 c.set_xticks(range(3),['balanced','3+5 split','comb']);c.set_ylim(0,.90);c.set_ylabel('mean best-restart NMSE',fontsize=PT_LABEL)
 c.text(.03,.96,'Same four input assignments per shape',transform=c.transAxes,va='top',fontsize=PT_SMALL)
 c.text(.03,.85,'◇ mean centered-cut bound',transform=c.transAxes,fontsize=PT_SMALL)
 d=fig.add_subplot(gs[1,1]);clean(d,'x');title(d,'D','Selection within twelve candidates')
 keys=['centered_cut_bound','full_cut_bound','centered_cut_sum','two_sweep_pilot','best_fixed_in_hindsight','uniform_random_expectation']
 names=['centered-cut bound','full-cut bound','sum of tails (heuristic)','two-sweep fitting pilot','best fixed in hindsight','random expectation']
 for i,(key,name) in enumerate(zip(keys,names)):
  for j,(fam,col) in enumerate(list(FC.items())[:2]):
   row=policy[policy.family.eq(fam)&policy.policy.eq(key)].iloc[0]
   d.plot(row.mean_regret,i+(j-.5)*.14,'o' if j==0 else '^',color=col,ms=3.5)
  d.text(.015,i-.28,name,transform=d.get_yaxis_transform(),fontsize=PT_SMALL)
 d.set_yticks([]);d.set_ylim(5.35,-.72);d.set_xlim(-.009,.29);d.set_xlabel('mean excess population NMSE',fontsize=PT_LABEL)
 fig.text(.5,.985,'Exact finite-domain diagnostic: 140 tasks × 12 candidates; all 256 inputs',ha='center',va='top',fontsize=PT_ANNOT)
 save(fig,36,'D',[STRUCT/'candidate_outcomes.csv',STRUCT/'audit_policy_summary.csv'])

def draw_tree(ax,tree,color):
 children={int(k):v for k,v in tree['children'].items()};pos={};order=[]
 def visit(v,depth):
  if v<8:pos[v]=(len(order),4-depth);order.append(v)
  else:
   for child in children[v]:visit(child,depth+1)
   pos[v]=(np.mean([pos[q][0] for q in children[v]]),4-depth)
 visit(14,0)
 for v,pair in children.items():
  for q in pair:ax.plot([pos[v][0],pos[q][0]],[pos[v][1],pos[q][1]],color='#7C878D',lw=LW_EDGE)
 for v,(x,y) in pos.items():
  ax.scatter([x],[y],marker='s' if v<8 else 'o',s=15 if v<8 else 24,facecolor='white' if v<8 else color,edgecolor=color,lw=LW_EDGE,zorder=3)
  if v<8:ax.text(x,y-.27,rf'$x_{{{v+1}}}$',ha='center',va='top',fontsize=PT_SMALL)
 ax.set_xlim(-.6,7.6);ax.set_ylim(-.75,4.6);ax.axis('off')

def s37():
 source=STRUCT/'constructive_dp_v2';end=pd.read_csv(source/'adaptive_constructions.csv')
 trees={t['task_id']:t for t in json.loads((source/'constructed_trees.json').read_text())}
 depth=pd.read_csv(STRUCT/'design/constructive_depth_certificate.csv')
 fig=plt.figure(figsize=(FIG_W,5.4));gs=fig.add_gridspec(2,2,left=.10,right=.97,top=.88,bottom=.13,wspace=.40,hspace=.52)
 for j,(tid,col,name) in enumerate([('matching_000',FC['quadratic_matching'],'Parallel: minimum depth 3'),('nested_prefix_000',FC['nested_prefix_control'],'Nested: minimum depth 4')]):
  ax=fig.add_subplot(gs[0,j]);draw_tree(ax,trees[tid],col);title(ax,'AB'[j],name)
  ax.text(.5,1.02,'root readout',color=col,transform=ax.transAxes,ha='center',fontsize=PT_SMALL)
 c=fig.add_subplot(gs[1,0]);clean(c,'y');title(c,'C','Depth constraint over all labeled trees')
 for fam,col in FC.items():
  z=depth[depth.family.eq(fam)].groupby('maximum_depth').best_centered_cut_bound.mean()
  c.plot(z.index,z,color=col,marker={'quadratic_matching':'o','quartic_partition':'s','nested_prefix_control':'^'}[fam],ms=3.5,lw=LW_DATA)
 c.set_xticks([3,4,5]);c.set_xlabel('maximum root-to-leaf edge depth',fontsize=PT_LABEL);c.set_ylabel('minimum cut bound (NMSE)',fontsize=PT_LABEL)
 c.set_ylim(-.018,.18);c.text(.36,.46,'Zero bound ⇔ exact fit\nin this multi-affine class',transform=c.transAxes,fontsize=PT_SMALL)
 d=fig.add_subplot(gs[1,1]);d.axis('off');title(d,'D','Exact construction at fixed resources')
 d.text(0,.94,'28 coefficients · 14 edges · 8 inputs',fontsize=PT_ANNOT)
 for y,fam,name in zip([.70,.51,.32],FC,['105 matching targets','35 quartic targets','24 nested controls']):
  z=end[end.family.eq(fam)]
  d.text(0,y,name,color=FC[fam],fontsize=PT_ANNOT)
  d.text(.96,y,f'depth {int(z.depth.iloc[0])}',ha='right',fontsize=PT_ANNOT)
 d.text(0,.12,r'All NMSE $<7\times10^{-30}$; max $|w|\simeq1$',fontsize=PT_ANNOT)
 d.text(0,-.09,'Primary spectra: (¼, …, ¼).\nNested: (¼, ¼, ½, ½, ¾, ¾, 1, 1).',fontsize=PT_SMALL,linespacing=1.4)
 fig.text(.5,.985,'Full-target Fourier construction; specific to nodes affine in each child',ha='center',va='top',fontsize=PT_ANNOT)
 save(fig,37,'D',[source/'adaptive_constructions.csv',source/'constructed_trees.json',STRUCT/'design/constructive_depth_certificate.csv'])

def s38():
 source=DYN/'summaries/fresh';summary=pd.read_csv(source/'endpoint_summary_with_secondary_baselines.csv')
 fig=plt.figure(figsize=(FIG_W,6.2));gs=fig.add_gridspec(2,2,left=.10,right=.97,top=.90,bottom=.10,wspace=.43,hspace=.42,height_ratios=[1.15,1])
 for j,arm in enumerate(ARMS):
  ax=fig.add_subplot(gs[0,j]);clean(ax,'x');title(ax,'AB'[j],ARM_NAMES[j]+': final selection')
  for y,method in enumerate(METHODS):
   row=summary[summary.arm.eq(arm)&summary.regime.eq('fixed_cache')&summary.method.eq(method)&summary['rank'].eq('all')].iloc[0]
   ax.errorbar(row.mean_regret,y,xerr=[[max(0,row.mean_regret-row.ci95_low)],[max(0,row.ci95_high-row.mean_regret)]],fmt='o',ms=3,color=MC[method],lw=LW_DATA,capsize=1.8)
   ax.text(.015,y-.29,NAMES[method],transform=ax.get_yaxis_transform(),fontsize=PT_SMALL,color=MC[method])
  ax.set_yticks([]);ax.set_ylim(7.35,-.75);ax.set_xlim(-.003,.12);ax.set_xlabel('costed test-loss regret',fontsize=PT_LABEL)
 c=fig.add_subplot(gs[1,0]);clean(c,'y');title(c,'C','Strong baselines in the joint arm')
 for j,method in enumerate(['gaussian_plugin_sgd','gaussian_plugin_fullbatch','observed_count_cheapest','gaussian_oracle_sgd']):
  z=summary[summary.arm.eq(ARMS[1])&summary.regime.eq('fixed_cache')&summary.method.eq(method)&summary['rank'].ne('all')].copy();z['r']=z['rank'].astype(int);z=z.sort_values('r')
  c.errorbar(np.arange(4)+(j-1.5)*.06,z.mean_regret*1000,yerr=[np.maximum(0,z.mean_regret-z.ci95_low)*1000,np.maximum(0,z.ci95_high-z.mean_regret)*1000],marker='o',ms=3,lw=LW_DATA,capsize=1.5,color=MC[method],label=NAMES[method])
 c.set_xticks(range(4),[1,2,4,8]);c.set_xlabel('generating rank',fontsize=PT_LABEL);c.set_ylabel(r'regret ($\times10^{-3}$)',fontsize=PT_LABEL)
 c.legend(loc='upper left',fontsize=PT_SMALL,frameon=False);c.set_ylim(top=c.get_ylim()[1]*1.5)
 d=fig.add_subplot(gs[1,1]);clean(d,'x');title(d,'D','Measured single-CPU cost')
 timing=pd.read_csv(source/'comparative_computational_cost.csv').set_index('method')
 keys=['observed_count_cheapest','gaussian_plugin_fullbatch','gaussian_plugin_sgd','empirical_split_fullbatch','original_scalar','pilot16','candidate_training']
 for y,key in enumerate(keys):
  d.plot(timing.loc[key,'median_seconds_per_20_candidates'],y,'o',color=MC.get(key,'#5A6A74'),ms=3.7)
  d.text(.01,y-.29,NAMES[key],transform=d.get_yaxis_transform(),fontsize=PT_SMALL)
 d.set_yticks([]);d.set_ylim(6.35,-.75);d.set_xscale('log');d.set_xlim(.001,.3);d.set_xlabel('seconds per 20-candidate decision',fontsize=PT_LABEL)
 fig.text(.5,.985,'Fresh seeds: 20 independent blocks; fixed-cache endpoint; 95% seed-bootstrap intervals',ha='center',va='top',fontsize=PT_ANNOT)
 save(fig,38,'D',[source/'endpoint_summary_with_secondary_baselines.csv',source/'comparative_computational_cost.csv'])

def s39():
 source=DYN/'summaries/fresh';runs=DYN/'runs/fresh'
 opaths=sorted(runs.glob('*_outcomes.csv'));ppaths=sorted(runs.glob('*_predictions.csv'))
 outcomes=pd.concat([pd.read_csv(p) for p in opaths],ignore_index=True)
 predictions=pd.concat([pd.read_csv(p) for p in ppaths],ignore_index=True)
 biases=pd.read_csv(source/'oracle_population_bias_by_regime.csv')
 fig=plt.figure(figsize=(FIG_W,6.8));gs=fig.add_gridspec(3,2,left=.11,right=.93,top=.92,bottom=.08,wspace=.48,hspace=.57)
 axes=[];hexes=[];key=['task_id','arm','candidate_id','checkpoint']
 for col,arm in enumerate(ARMS):
  actual=outcomes[outcomes.arm.eq(arm)&outcomes.regime.eq('fixed_cache')&outcomes.checkpoint.eq(256)]
  for row,method in enumerate(['original_scalar','gaussian_plugin_sgd']):
   z=actual.merge(predictions[predictions.method.eq(method)][key+['predicted_loss']],on=key,validate='one_to_one')
   ax=fig.add_subplot(gs[row,col]);axes.append(ax);clean(ax)
   title(ax,chr(65+2*row+col),ARM_NAMES[col]+(': scalar' if row==0 else ': finite-horizon'))
   hb=ax.hexbin(z.predicted_loss,z.test_loss,gridsize=23,mincnt=1,cmap='viridis',norm=LogNorm(vmin=1),linewidths=0);hexes.append(hb)
   lo=min(z.predicted_loss.min(),z.test_loss.min());hi=max(z.predicted_loss.max(),z.test_loss.max())
   ax.plot([lo,hi],[lo,hi],ls='--',lw=LW_REF,color='#C06B57');ax.set_xlabel('predicted final half-MSE',fontsize=PT_SMALL);ax.set_ylabel('observed test half-MSE',fontsize=PT_SMALL)
  ax=fig.add_subplot(gs[2,col]);clean(ax,'y');title(ax,'EF'[col],ARM_NAMES[col]+': cache sensitivity')
  for j,regime in enumerate(['fixed_cache','fresh_iid']):
   z=biases[biases.arm.eq(arm)&biases.regime.eq(regime)&biases.checkpoint.eq(256)&biases.noise_sd.eq(.75)].sort_values('rank')
   ax.errorbar(np.arange(4)+(j-.5)*.10,z.mean_prediction_minus_population_loss*1000,yerr=[(z.mean_prediction_minus_population_loss-z.ci95_low)*1000,(z.ci95_high-z.mean_prediction_minus_population_loss)*1000],fmt='o-',ms=3,capsize=1.5,lw=LW_DATA,label='reused cache' if j==0 else 'fresh examples',color=[COLORS['local'],COLORS['shunting']][j])
  ax.axhline(0,color='#7A858B',lw=LW_REF);ax.set_xticks(range(4),[1,2,4,8]);ax.set_xlabel('generating rank',fontsize=PT_SMALL)
  ax.set_ylabel('oracle minus population loss\n'+r'($\times10^{-3}$)',fontsize=PT_SMALL);ax.legend(frameon=False,fontsize=PT_SMALL,loc='lower left')
 vmax=max(float(h.get_array().max()) for h in hexes)
 for h in hexes:h.set_norm(LogNorm(vmin=1,vmax=vmax))
 ca=fig.add_axes([.953,.53,.013,.25]);bar=fig.colorbar(hexes[0],cax=ca);bar.ax.tick_params(labelsize=PT_SMALL,length=2);bar.ax.set_title('count',fontsize=PT_SMALL,pad=6)
 fig.text(.5,.988,'Endpoint prediction and the fresh-example assumption',ha='center',va='top',fontsize=PT_LABEL)
 save(fig,39,'F',opaths+ppaths+[source/'oracle_population_bias_by_regime.csv'])

def main():
 apply_neurips_style();OUT.mkdir(parents=True,exist_ok=True)
 # Explicit braces are needed by Matplotlib's mathtext parser.
 s36();s37();s38();s39()

if __name__=='__main__':main()
