#!/usr/bin/env python3
"""Native Boolean morphology illustrations and frozen-data supplementary plots.

The canonical-gate illustration concerns one coordinate representation. It does
not assert that broadcast training cannot find a different representation.
Experiment outcomes are read only after the independent analysis is complete.
"""
from pathlib import Path
import argparse
import ast,re
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import ListedColormap,LinearSegmentedColormap,LogNorm
from build_morphology_followup_figures import clean,title,save
from journal_style import FIG_W,PT_SMALL,PT_LABEL,PT_ANNOT,LW_DATA,LW_REF,apply_neurips_style

ROOT=Path(__file__).resolve().parents[1]
THEORY=ROOT/'source_data/boolean_theory'
LEARNING=ROOT/'source_data/boolean_morphology'
OUT=ROOT/'figures/supplementary'
PREVIEW=ROOT/'analysis/boolean_morphology_revision_20260905'
GREEN='#278365'
ORANGE='#C27B26'
GRAY='#707B84'
PURPLE='#8055AD'


def gate_box(ax,x,y,label,color,width=.09):
 ax.add_patch(FancyBboxPatch((x-width/2,y-.039),width,.078,
                            boxstyle='round,pad=.012',lw=.8,
                            ec=color,fc='white',zorder=4))
 ax.text(x,y,label,ha='center',va='center',fontsize=PT_SMALL-1,color=color,zorder=5)


def grouping_schematic(ax):
 """Four-input XOR-of-AND example, sized for the existing main Figure6A."""
 capacity=pd.read_csv(THEORY/'main6a_grouping.csv').set_index('grouping')
 for j,(labels,color,name) in enumerate([
     ([1,2,3,4],GREEN,'aligned grouping'),
     ([1,3,2,4],GRAY,'crossed grouping')]):
  left=j*.52
  leaves=left+np.array([.045,.165,.315,.435])
  children=left+np.array([.105,.375])
  root=left+.24
  for x in children:ax.plot([x,root],[.59,.83],color=color,lw=.85,zorder=1)
  for k,x in enumerate(leaves):
   ax.plot([x,children[k//2]],[.36,.59],color=color,lw=.85,zorder=1)
   ax.plot(x,.36,marker='s',ms=3.2,mfc='white',mec=color,mew=.8,zorder=3)
   ax.text(x,.285,rf'$b_{labels[k]}$',ha='center',fontsize=PT_SMALL)
  gate_box(ax,root,.83,'XOR' if j==0 else r'$u_3$',color,width=.10)
  for k,x in enumerate(children):gate_box(ax,x,.59,'AND' if j==0 else rf'$u_{k+1}$',color,width=.10)
  if j==0:
   ax.text(children[0]+.04,.665,r'$u$',fontsize=PT_SMALL,color=color)
   ax.text(children[1]-.075,.665,r'$v$',fontsize=PT_SMALL,color=color)
  ax.text(root,.965,name,ha='center',fontsize=PT_SMALL,color=color)
  key=['ab|cd','ac|bd'][j]
  ax.text(root,.19,capacity.loc[key,'display_label'],ha='center',fontsize=PT_SMALL,color=color)
 ax.text(.5,.095,r'Canonical gate coordinates: $\partial y/\partial u=1-2v$',ha='center',fontsize=PT_SMALL)
 ax.text(.5,-.003,'12 coefficients · 6 edges · all local units learnable',ha='center',fontsize=PT_SMALL)
 ax.set_xlim(-.025,1.025);ax.set_ylim(-.04,1.04);ax.axis('off')


FAMILIES=['and4','or4','parity4','or_of_ands','xor_of_ands','and_of_xors','nested']
FAMILY_NAMES=['AND','OR','parity','OR(AND)','XOR(AND)','AND(XOR)','nested']
FAMILY_LABEL=dict(zip(FAMILIES,FAMILY_NAMES))


def parse_tree(text):
 return ast.literal_eval(re.sub(r'[abcd]',lambda m:repr(m.group(0)),text))


def simple_tree(ax,tree,color):
 leaves=[]
 def walk(node):
  if isinstance(node,str):leaves.append(node)
  else:walk(node[0]);walk(node[1])
 walk(tree)
 pos={v:(k+.5)/4 for k,v in enumerate(leaves)}
 def draw(node,depth=0):
  y=1-depth*.26
  if isinstance(node,str):
   x=pos[node];ax.plot(x,y,'s',ms=3,mfc='white',mec=color,mew=.8)
   ax.text(x,y-.12,node,ha='center',fontsize=PT_SMALL)
  else:
   points=[draw(n,depth+1) for n in node];x=np.mean([p[0] for p in points])
   for xx,yy in points:ax.plot([x,xx],[y,yy],color=color,lw=.9,zorder=1)
   ax.plot(x,y,'o',ms=4,color=color,zorder=2)
  return x,y
 draw(tree);ax.set_xlim(0,1);ax.set_ylim(-.01,1.13);ax.axis('off')


def s43():
 truth=pd.read_csv(THEORY/'truth_tables.csv')
 trees=pd.read_csv(THEORY/'tree_capacity.csv')
 summary=pd.read_csv(THEORY/'target_summary.csv').set_index('family').loc[FAMILIES]
 gates=pd.read_csv(THEORY/'gate_derivative_corners.csv')
 grid=pd.read_csv(THEORY/'gate_credit_fields.csv')
 fig=plt.figure(figsize=(FIG_W,6.65))
 gs=fig.add_gridspec(3,2,left=.12,right=.93,top=.92,bottom=.08,wspace=.54,hspace=.78)
 a=fig.add_subplot(gs[0,0]);title(a,'A','Seven exact Boolean truth tables')
 matrix=truth.pivot(index='family',columns='pattern_index',values='target_raw').loc[FAMILIES]
 a.imshow(matrix,aspect='auto',cmap=ListedColormap(['#F6F8F7',GREEN]),vmin=0,vmax=1,interpolation='nearest')
 bits=truth[truth.family.eq(FAMILIES[0])].sort_values('pattern_index')
 bitnames=[''.join(str(int(r[k])) for k in ['a','b','c','d']) for _,r in bits.iterrows()]
 a.set_xticks(range(16),bitnames,rotation=90,fontsize=PT_SMALL);a.set_yticks(range(7),FAMILY_NAMES,fontsize=PT_SMALL)
 a.tick_params(length=0);a.set_xlabel('input bits abcd; white 0, green 1',fontsize=PT_SMALL)
 a.spines[['top','right','left','bottom']].set_visible(False)
 b=fig.add_subplot(gs[0,1]);title(b,'B','All trees: regression obstruction')
 definitions=trees.drop_duplicates('tree_id').sort_values(['depth','tree_id'])
 order=definitions.tree_id.to_list()
 bound=trees.pivot(index='family',columns='tree_id',values='normalized_mse_lower_bound').loc[FAMILIES,order]
 cmap=LinearSegmentedColormap.from_list('boolean_bound',['#F9FBFA','#C7DAD0','#C27B26'])
 im=b.imshow(bound,aspect='auto',cmap=cmap,vmin=0,vmax=float(bound.to_numpy().max()),interpolation='nearest')
 yy,xx=np.where(bound.to_numpy()<1e-12);b.scatter(xx,yy,s=4,color=GREEN,lw=0)
 b.set_xticks(range(15),order,rotation=90,fontsize=PT_SMALL);b.set_yticks(range(7),FAMILY_NAMES,fontsize=PT_SMALL);b.tick_params(length=0)
 b.axvline(2.5,color='#69747B',lw=.75);b.spines[['top','right','left','bottom']].set_visible(False)
 b.set_xlabel('3 balanced | 12 comb; dots: exact possible',fontsize=PT_SMALL)
 cb=fig.colorbar(im,ax=b,fraction=.046,pad=.02);cb.set_label('NMSE bound',fontsize=PT_SMALL);cb.ax.tick_params(labelsize=PT_SMALL)
 c=fig.add_subplot(gs[1,0]);c.axis('off');title(c,'C','Equal resources; different minimum depth')
 for j,(family,color) in enumerate([('or_of_ands',GREEN),('nested',ORANGE)]):
  row=trees[trees.family.eq(family)&trees.zero_bound_exact].sort_values(['depth','tree_id']).iloc[0]
  sub=c.inset_axes([.02+j*.51,.11,.46,.77]);simple_tree(sub,parse_tree(row.tree),color)
  sub.text(.5,1.05,FAMILY_LABEL[family],transform=sub.transAxes,ha='center',color=color,fontsize=PT_SMALL)
  c.text(.25+j*.51,.06,f'minimum depth {int(row.depth)}',ha='center',fontsize=PT_SMALL,color=color)
 c.text(.5,-.13,'12 coefficients and 6 edges for every tree',ha='center',fontsize=PT_SMALL)
 d=fig.add_subplot(gs[1,1]);clean(d,'y');title(d,'D','Associative gates are structure controls')
 d.scatter(range(7),summary.minimum_exact_depth,s=14,color=GREEN,zorder=3)
 for i,(_,r) in enumerate(summary.iterrows()):d.text(i,r.minimum_exact_depth+.105,f'{int(r.exact_compatible_trees)}/15',ha='center',fontsize=PT_SMALL)
 d.set_xticks(range(7),FAMILY_NAMES,rotation=50,ha='right');d.set_yticks([2,3]);d.set_ylim(1.75,3.42);d.set_ylabel('minimum exact depth',fontsize=PT_LABEL)
 d.text(.02,.95,'Labels: exact-compatible trees',va='top',transform=d.transAxes,fontsize=PT_SMALL)
 e=fig.add_subplot(gs[2,0]);clean(e,'y');title(e,'E','Post hoc: target information in a pair')
 energy=pd.read_csv(THEORY/'proper_subtree_projection_energy.csv')
 subsets=['ab','ac','ad','bc','bd','cd']
 for family,color,label in [('xor_of_ands',GREEN,'XOR(AND)'),('parity4',PURPLE,'parity')]:
  z=energy[energy.family.eq(family)&energy.subset_size.eq(2)].set_index('subset').loc[subsets]
  e.plot(range(6),z.projection_energy_normalized,marker='o',ms=3,color=color,lw=LW_DATA,label=label)
 e.set_xticks(range(6),subsets);e.set_ylim(-.025,.265);e.set_yticks([0,.1,.2]);e.set_xlabel('all six input pairs',fontsize=PT_LABEL);e.set_ylabel('normalized target-\nprojection energy',fontsize=PT_LABEL)
 e.legend(frameon=False,fontsize=PT_SMALL,loc='center left')
 f=fig.add_subplot(gs[2,1]);clean(f,'y');title(f,'F','Canonical conditional credit')
 for gate,color in zip(['AND','OR','XOR'],[GREEN,ORANGE,PURPLE]):
  z=grid[grid.gate.eq(gate)&grid.left.eq(0)].sort_values('right')
  f.plot(z.right,z.d_output_d_left,color=color,lw=LW_DATA,label=gate)
 f.axhline(0,color='#A0A9AE',lw=LW_REF,ls=':');f.set_xlim(-.025,1.025);f.set_ylim(-1.1,1.1);f.set_xticks([0,.5,1]);f.set_yticks([-1,0,1]);f.set_xlabel('other branch output v',fontsize=PT_LABEL);f.set_ylabel(r'$\partial F/\partial u$',fontsize=PT_LABEL)
 f.legend(frameon=False,fontsize=PT_SMALL,loc='lower left')
 save(fig,43,'F',[THEORY/x for x in ['truth_tables.csv','tree_capacity.csv','target_summary.csv','gate_derivative_corners.csv','gate_credit_fields.csv','exact_constructions.json','proper_subtree_projection_energy.csv','report.json']])


LEARNING_TREES=['balanced_ab_cd','balanced_ac_bd','balanced_ad_bc','comb_a_b_cd']
TREE_LABELS=['ab|cd','ac|bd','ad|bc','a|(b|cd)']


def s44():
 summary=pd.read_csv(LEARNING/'condition_summary.csv')
 end=pd.read_csv(LEARNING/'selected_endpoints.csv')
 primary=pd.read_csv(LEARNING/'primary_contrasts.csv')
 pairs=pd.read_csv(LEARNING/'paired_primary_contrasts.csv')
 same=pd.read_csv(LEARNING/'same_rate_contrast_summary.csv')
 trajectory=pd.read_csv(LEARNING/'trajectory_summary.csv')
 rates=json.loads((LEARNING/'selected_rates.json').read_text())
 assert len(summary)==112 and len(end)==2240
 values=summary.mean_population_nmse.to_numpy()
 assert np.isfinite(values).all() and np.min(values)>=0
 positive=values[values>0]
 vmin=10**np.floor(np.log10(positive.min())) if len(positive) else 1e-8
 vmax=10**np.ceil(np.log10(max(1.,values.max())))
 norm=LogNorm(vmin=vmin,vmax=vmax)
 fig=plt.figure(figsize=(FIG_W,6.85))
 gs=fig.add_gridspec(4,2,left=.13,right=.92,top=.94,bottom=.075,wspace=.64,hspace=.72)
 for index,(opt,rule) in enumerate([('adam','exact'),('adam','broadcast'),('sgd','exact'),('sgd','broadcast')]):
  ax=fig.add_subplot(gs[index//2,index%2]);title(ax,'ABCD'[index],f'{opt.upper()}: {"exact" if rule=="exact" else "broadcast"}; rate {rates[opt][rule]:g}')
  z=summary[summary.optimizer.eq(opt)&summary.rule.eq(rule)].pivot(index='family',columns='tree',values='mean_population_nmse').loc[FAMILIES,LEARNING_TREES]
  im=ax.imshow(np.maximum(z,vmin),aspect='auto',interpolation='nearest',cmap='viridis',norm=norm)
  ax.set_xticks(range(4),TREE_LABELS,fontsize=PT_SMALL);ax.set_yticks(range(7),FAMILY_NAMES,fontsize=PT_SMALL);ax.tick_params(length=0)
  ax.spines[['top','right','left','bottom']].set_visible(False)
  if index==1:
   cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.04);cb.ax.tick_params(labelsize=PT_SMALL);cb.set_label('clean NMSE (A–D)',fontsize=PT_SMALL)
 e=fig.add_subplot(gs[2,0]);e.axis('off');title(e,'E','Both primary tests, including the margin')
 keys=['crossed_minus_compatible_exact','broadcast_minus_exact_compatible']
 for j,(key,label,color) in enumerate(zip(keys,['grouping','credit'],[GREEN,GRAY])):
  sub=e.inset_axes([j*.61,.04,.35,.82]);clean(sub,'y')
  r=primary[primary.contrast.eq(key)].iloc[0];z=pairs[pairs.contrast.eq(key)].sort_values('seed')
  sub.scatter(np.linspace(-.15,.15,20),z.difference,s=5,color=color,alpha=.35)
  sub.errorbar(0,r.mean_difference,yerr=[[r.mean_difference-r.ci975_low],[r.ci975_high-r.mean_difference]],fmt='D',ms=3.2,color=color,capsize=1.5,lw=LW_DATA)
  sub.axhline(.01,ls=':',lw=LW_REF,color=ORANGE);sub.set_xticks([]);sub.set_xlim(-.3,.3)
  sub.set_title(label,fontsize=PT_SMALL,pad=4)
  if j==0:sub.set_ylim(0,max(.67,float(z.difference.max())*1.1));sub.set_ylabel('NMSE difference',fontsize=PT_SMALL)
  else:sub.set_ylim(-.0005,.0125);sub.set_yticks([0,.005,.01],['0','.005','.010'])
  sub.text(.5,-.14,'passes' if bool(r.passes_adjusted_interval_and_mean_margin) else 'below margin',ha='center',transform=sub.transAxes,color=color,fontsize=PT_SMALL)
 f=fig.add_subplot(gs[2,1]);clean(f,'y');title(f,'F','Classification and regression differ')
 for metric,color,marker,label in [('mean_accuracy','#9CA6AA','o','accuracy'),('mean_balanced_accuracy',GREEN,'x','balanced accuracy')]:
  f.scatter(summary.mean_population_nmse,summary[metric],s=8,color=color,marker=marker,alpha=.55,label=label,lw=.6)
 f.scatter([1,1],[15/16,.5],marker='*',s=28,color=ORANGE,zorder=5)
 f.set_xscale('log');f.set_xlim(vmin*.8,vmax*1.2);f.set_ylim(-.025,1.05);f.set_xlabel('clean population NMSE',fontsize=PT_SMALL);f.set_ylabel('threshold performance',fontsize=PT_SMALL)
 f.legend(frameon=False,fontsize=PT_SMALL,loc='lower left')
 g=fig.add_subplot(gs[3,0]);g.axis('off');title(g,'G','Same-rate XOR(AND) credit controls')
 for j,(opt,color) in enumerate([('adam',GREEN),('sgd',ORANGE)]):
  sub=g.inset_axes([j*.61,.07,.35,.83]);clean(sub,'y')
  z=same[same.optimizer.eq(opt)&same.contrast.eq('broadcast_minus_exact_compatible')].sort_values('rate')
  sub.errorbar(np.arange(3),z.mean_difference,yerr=[z.mean_difference-z.ci95_low,z.ci95_high-z.mean_difference],fmt='o-',ms=2.4,lw=LW_DATA,color=color,capsize=1.2)
  sub.axhline(0,color='#A0A9AE',lw=LW_REF);sub.axhline(.01,color='#A0A9AE',lw=LW_REF,ls=':')
  sub.set_xticks(range(3),['.003','.01','.03'],fontsize=PT_SMALL,rotation=45);sub.set_title(opt.upper(),fontsize=PT_SMALL,pad=4);sub.set_xlabel('common rate',fontsize=PT_SMALL)
  sub.set_xlim(-.3,2.3)
  if j==0:sub.set_ylim(-.0008,.0115);sub.set_yticks([0,.005,.01],['0','.005','.010']);sub.set_ylabel('broadcast − exact NMSE',fontsize=PT_SMALL)
  else:sub.set_ylim(min(-.1,z.ci95_low.min()*1.15),max(.2,z.ci95_high.max()*1.15));sub.set_yticks([-.1,0,.1,.2])
 h=fig.add_subplot(gs[3,1]);clean(h,'y');title(h,'H','Broadcast: gradients at own trained states')
 for family,color,label in [('xor_of_ands',GREEN,'XOR(AND)'),('parity4',PURPLE,'parity')]:
  for opt,ls in [('adam','-'),('sgd','--')]:
   z=trajectory[trajectory.family.eq(family)&trajectory.optimizer.eq(opt)&trajectory.rule.eq('broadcast')&trajectory.tree.eq('balanced_ab_cd')].sort_values('step')
   h.plot(z.step,z.mean_gradient_cosine,color=color,ls=ls,lw=LW_DATA,label=label+' '+opt.upper())
   h.fill_between(z.step,z.gradient_cosine_ci95_low,z.gradient_cosine_ci95_high,color=color,alpha=.07,lw=0)
 h.axhline(1,color='#A0A9AE',lw=LW_REF,ls=':');h.set_xscale('symlog',linthresh=1);h.set_xlim(-.1,2400);h.set_xticks([0,1,16,256,2048],[0,1,16,256,2048]);h.set_ylim(min(-.25,float(trajectory[trajectory.rule.eq('broadcast')&trajectory.family.isin(['xor_of_ands','parity4'])&trajectory.tree.eq('balanced_ab_cd')].gradient_cosine_ci95_low.min())-.05),1.05);h.set_ylabel('population-gradient\ncosine',fontsize=PT_SMALL);h.set_xlabel('training update',fontsize=PT_SMALL)
 h.legend(frameon=False,fontsize=PT_SMALL,loc='lower left',ncol=2)
 save(fig,44,'H',[LEARNING/x for x in ['condition_summary.csv','selected_endpoints.csv','primary_contrasts.csv','paired_primary_contrasts.csv','same_rate_contrast_summary.csv','trajectory_summary.csv','selected_rates.json','protocol.json','analysis_record.json']])


def preview():
 apply_neurips_style()
 preview_width=(FIG_W-(43+13+37)/72)/2
 fig,ax=plt.subplots(figsize=(preview_width,1.8))
 fig.subplots_adjust(left=.005,right=.995,top=.99,bottom=.005)
 grouping_schematic(ax)
 PREVIEW.mkdir(exist_ok=True)
 path=PREVIEW/'proposed_main6A_boolean.pdf'
 fig.savefig(path);fig.savefig(path.with_suffix('.png'),dpi=220);plt.close(fig)
 print(path)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--preview-only',action='store_true')
 parser.add_argument('--theory-only',action='store_true')
 parser.add_argument('--learning-only',action='store_true')
 args=parser.parse_args()
 if args.preview_only:preview();return
 apply_neurips_style()
 if not args.learning_only:s43()
 if not args.theory_only:s44()

if __name__=='__main__':main()
