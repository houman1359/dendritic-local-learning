#!/usr/bin/env python3
"""Publication plots for independently completed morphology bridge studies."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from journal_style import COLORS,style_direct_color_labels,FIG_W,PT_SMALL,PT_ANNOT,PT_LABEL,LW_DATA,LW_REF,apply_neurips_style
from build_morphology_followup_figures import clean,title,save

ROOT=Path(__file__).resolve().parents[1]
CREDIT=ROOT/'source_data/morphology_credit'
CALIB=ROOT/'source_data/morphology_calibration'
COND=ROOT/'source_data/morphology_conductance'
RULES=['exact','broadcast','global_projection','subtree_projection','shuffled_projection']
RC={'exact':COLORS['shunting'],'broadcast':COLORS['local'],'global_projection':'#648C9B','subtree_projection':COLORS['oracle'],'shuffled_projection':COLORS['mute']}
RN={'exact':'exact path','broadcast':'root broadcast','global_projection':'one oracle profile','subtree_projection':'two subtree profiles','shuffled_projection':'two shuffled profiles'}
FAMILIES=['matching','quartet','nested']

def credit_families(ax,optimizer='adam',rules=RULES,legend=False):
 summary=pd.read_csv(CREDIT/'figure_condition_summary.csv')
 for j,rule in enumerate(rules):
  z=summary[summary.structure.eq('compatible')&summary.optimizer.eq(optimizer)&summary.rule.eq(rule)].set_index('family').loc[FAMILIES]
  offset=(j-(len(rules)-1)/2)*(.075 if len(rules)>2 else .14)
  ax.errorbar(np.arange(3)+offset,z['mean'],yerr=[z['mean']-z.ci95_low,z.ci95_high-z['mean']],fmt='o',ms=3.1,color=RC[rule],lw=LW_DATA,capsize=1.5,label=RN[rule])
 for x,floor in enumerate([.0225,.045,.0225]):ax.plot([x-.25,x+.25],[floor,floor],color='#A2AAAF',lw=LW_REF,ls=':')
 upper=summary[summary.structure.eq('compatible')&summary.rule.isin(rules)].ci95_high.max()
 ax.set_xticks(range(3),['matching','quartic','nested']);ax.set_ylim(-.025,max(1.08,float(upper)*1.04));ax.set_ylabel('test NMSE',fontsize=PT_LABEL)
 if legend:ax.legend(frameon=False,fontsize=PT_SMALL,loc='upper left')

def s40():
 source=CREDIT/'summaries/fresh'
 contrasts=pd.read_csv(source/'paired_contrasts.csv')
 gradient=pd.read_csv(CREDIT/'figure_gradient_summary.csv')
 rates=pd.read_csv(source/'all_rate_sensitivity.csv')
 selection=json.loads((CREDIT/'development_fit.json').read_text())
 fig=plt.figure(figsize=(FIG_W,6.75))
 gs=fig.add_gridspec(3,2,left=.10,right=.97,top=.865,bottom=.085,wspace=.42,hspace=.57)
 for j,opt in enumerate(['adam','sgd']):
  ax=fig.add_subplot(gs[0,j]);clean(ax,'y');title(ax,'AB'[j],opt.upper()+': compatible trees')
  credit_families(ax,opt)
 c=fig.add_subplot(gs[1,0]);clean(c,'y');title(c,'C','Exact credit: input assignment matters')
 for j,opt in enumerate(['adam','sgd']):
  z=contrasts[contrasts.optimizer.eq(opt)&contrasts.structure.eq('paired')&contrasts.contrast.eq('shuffled minus compatible: exact')&contrasts.family.isin(FAMILIES)].set_index('family').loc[FAMILIES]
  c.errorbar(np.arange(3)+(j-.5)*.14,z['mean'],yerr=[z['mean']-z.ci95_low,z.ci95_high-z['mean']],fmt='o',ms=3.2,lw=LW_DATA,capsize=1.8,color=[COLORS['shunting'],COLORS['local']][j],label=opt.upper())
 c.axhline(0,color='#929CA3',lw=LW_REF);c.set_xticks(range(3),['matching','quartic','nested']);c.set_ylabel('shuffled − compatible\nNMSE',fontsize=PT_LABEL);c.legend(frameon=False,fontsize=PT_SMALL)
 d=fig.add_subplot(gs[1,1]);clean(d,'y');title(d,'D','Adam: aggregate gradient alignment')
 for rule in RULES:
  z=gradient[gradient.structure.eq('compatible')&gradient.optimizer.eq('adam')&gradient.rule.eq(rule)].sort_values('step')
  d.plot(z.step,z['mean'],marker='o',ms=2.5,lw=LW_DATA,color=RC[rule]);d.fill_between(z.step,z.ci95_low,z.ci95_high,color=RC[rule],alpha=.12,lw=0)
 d.set_xscale('symlog',linthresh=1);d.set_xticks([0,1,16,256,1024],[0,1,16,256,1024]);d.set_ylim(-1.05,1.05);d.set_xlabel('training update',fontsize=PT_LABEL);d.set_ylabel('population-gradient\ncosine',fontsize=PT_LABEL)
 for j,opt in enumerate(['adam','sgd']):
  ax=fig.add_subplot(gs[2,j]);clean(ax,'y');title(ax,'EF'[j],opt.upper()+': all learning rates retained')
  for rule in RULES:
   z=rates[rates.optimizer.eq(opt)&rates.rule.eq(rule)].groupby('rate').test_nmse.mean().sort_index()
   ax.plot(z.index,z,marker='o',ms=2.7,lw=LW_DATA,color=RC[rule])
   chosen=selection[opt][rule];ax.plot(chosen,z.loc[chosen],marker='*',ms=6.4,mfc=RC[rule],mec='white',mew=.4)
  ax.set_xscale('log');ax.set_xticks([.003,.01,.03],['.003','.01','.03']);ax.set_xlabel('learning rate',fontsize=PT_LABEL);ax.set_ylabel('mean test NMSE',fontsize=PT_LABEL)
 handles=[Line2D([],[],color=RC[r],marker='o',ms=3,lw=LW_DATA,label=RN[r]) for r in RULES]
 fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.53,.995),frameon=False,fontsize=PT_SMALL)
 save(fig,40,'F',[CREDIT/'figure_condition_summary.csv',CREDIT/'figure_gradient_summary.csv',source/'paired_contrasts.csv',source/'all_rate_sensitivity.csv',CREDIT/'development_fit.json'])

POLICIES=['estimated_cut','development_best_fixed','two_sweep_pilot','uniform_random_expectation']
PC={'estimated_cut':COLORS['shunting'],'development_best_fixed':COLORS['point_mlp'],'two_sweep_pilot':COLORS['local'],'uniform_random_expectation':COLORS['mute']}
PN={'estimated_cut':'estimated interactions','development_best_fixed':'fixed / estimated-rank','two_sweep_pilot':'two-sweep pilot','uniform_random_expectation':'random expectation'}

END_FAMILIES=['matching','quartet','nested_prefix','random_interactions']
END_STRUCTURES=['estimated_dp','development_fixed','oracle_dp']
END_COLORS={'estimated_dp':COLORS['shunting'],'development_fixed':COLORS['point_mlp'],'oracle_dp':COLORS['oracle']}
END_NAMES={'estimated_dp':'estimated tree','development_fixed':'development-fixed tree','oracle_dp':'target-informed tree'}

def pipeline_structure(ax,legend=True):
 source=pd.read_csv(CALIB/'end_to_end/summary.csv')
 for j,structure in enumerate(END_STRUCTURES):
  z=source[source.structure.eq(structure)&source.rule.eq('exact')&source.family.isin(END_FAMILIES)].set_index('family').loc[END_FAMILIES]
  ax.errorbar(np.arange(4)+(j-1)*.12,z.mean_test_nmse,yerr=[z.mean_test_nmse-z.ci95_low,z.ci95_high-z.mean_test_nmse],fmt='o',ms=3.1,lw=LW_DATA,capsize=1.5,color=END_COLORS[structure],label=END_NAMES[structure])
 ax.set_xticks(range(4),['matching','quartic','nested','random']);ax.set_ylim(-.025,1.12);ax.set_ylabel('test NMSE',fontsize=PT_LABEL)
 if legend:ax.legend(frameon=False,fontsize=PT_SMALL,loc='upper left')

def pipeline_credit(ax,legend=True):
 source=pd.read_csv(CALIB/'end_to_end/summary.csv')
 for j,rule in enumerate(['exact','broadcast']):
  z=source[source.structure.eq('estimated_dp')&source.rule.eq(rule)&source.family.isin(END_FAMILIES)].set_index('family').loc[END_FAMILIES]
  ax.errorbar(np.arange(4)+(j-.5)*.16,z.mean_test_nmse,yerr=[z.mean_test_nmse-z.ci95_low,z.ci95_high-z.mean_test_nmse],fmt='o',ms=3.1,lw=LW_DATA,capsize=1.5,color=RC[rule],label=RN[rule])
 for x in range(4):ax.plot([x-.23,x+.23],[.0225,.0225],ls=':',lw=LW_REF,color='#A2AAAF')
 ax.set_xticks(range(4),['matching','quartic','nested','random']);ax.set_ylim(-.025,1.15);ax.set_ylabel('test NMSE',fontsize=PT_LABEL)
 if legend:ax.legend(frameon=False,fontsize=PT_SMALL,loc='upper left',bbox_to_anchor=(0,.72))

def end_to_end_panels(fig,gs):
 source=pd.read_csv(CALIB/'end_to_end/summary.csv')
 g=fig.add_subplot(gs[3,0]);clean(g,'y');title(g,'G','Fresh pipeline: all six conditions')
 for j,structure in enumerate(END_STRUCTURES):
  for k,rule in enumerate(['exact','broadcast']):
   z=source[source.structure.eq(structure)&source.rule.eq(rule)&source.family.isin(END_FAMILIES)].set_index('family').loc[END_FAMILIES]
   g.errorbar(np.arange(4)+(j-1)*.16+(k-.5)*.055,z.mean_test_nmse,yerr=[z.mean_test_nmse-z.ci95_low,z.ci95_high-z.mean_test_nmse],fmt='o' if k==0 else 's',ms=2.6,lw=LW_DATA,capsize=1,color=END_COLORS[structure],mfc=END_COLORS[structure] if k==0 else 'white')
 g.set_xticks(range(4),['matching','quartic','nested','random']);g.set_ylim(-.03,1.17);g.set_ylabel('test NMSE',fontsize=PT_LABEL)
 key_labels=['estimated\ntree','development-fixed\ntree','target-informed\ntree']
 handles=[Line2D([],[],marker='o',color=END_COLORS[t],ls='',ms=3,label=label) for t,label in zip(END_STRUCTURES,key_labels)]
 fig.legend(handles=handles,frameon=False,fontsize=PT_SMALL,ncol=3,loc='lower left',bbox_to_anchor=(.11,.003),borderaxespad=0,handlelength=.7,handletextpad=.3,columnspacing=.8)
 g.text(.03,.94,'filled: exact; open: root broadcast',transform=g.transAxes,fontsize=PT_SMALL)
 h=fig.add_subplot(gs[3,1]);clean(h,'x');title(h,'H','Fresh pipeline: pooled paired contrasts')
 cs=pd.read_csv(CALIB/'end_to_end/contrasts.csv')
 keys=['fixed_exact_minus_estimated_exact','estimated_broadcast_minus_exact','estimated_exact_minus_oracle_exact']
 labels=['fixed − estimated (exact credit)','broadcast − exact (estimated tree)','estimated − target-informed (exact)']
 for j,(key,label) in enumerate(zip(keys,labels)):
  r=cs[cs.family.eq('all')&cs.contrast.eq(key)].iloc[0];lo,hi=(r.ci975_low,r.ci975_high) if j<2 else (r.ci95_low,r.ci95_high)
  h.errorbar(r.mean_difference,j,xerr=[[r.mean_difference-lo],[hi-r.mean_difference]],fmt='o',ms=3,color=[COLORS['point_mlp'],COLORS['local'],COLORS['oracle']][j],lw=LW_DATA,capsize=1.7)
  h.text(.02,j-.27,label,transform=h.get_yaxis_transform(),fontsize=PT_SMALL)
 h.axvline(0,color='#929BA2',lw=LW_REF);h.set_ylim(2.45,-.65);h.set_yticks([]);h.set_xlabel('paired NMSE difference',fontsize=PT_LABEL)

def s41():
 source=pd.read_csv(CALIB/'policy_summary.csv')
 absolute=pd.read_csv(CALIB/'figure_absolute_error_summary.csv')
 contrasts=pd.read_csv(CALIB/'primary_contrasts.csv')
 records=pd.read_csv(CALIB/'calibration_selection_records.csv')
 # The final two panels are added only when the separate fresh cohort is complete.
 end_path=CALIB/'end_to_end/summary.csv'
 has_end=end_path.exists()
 rows=4 if has_end else 3
 fig=plt.figure(figsize=(FIG_W,6.55 if has_end else 5.65))
 gs=fig.add_gridspec(rows,2,left=.10,right=.97,top=.92,bottom=.07,wspace=.47,hspace=.72)
 a=fig.add_subplot(gs[0,0]);a.axis('off');title(a,'A','Finite labels determine the selected tree')
 for y,text in [(1,'256 labels: 192 fit + 64 pilot gate'),(.7,'Estimate interactions OR fit twelve pilots'),(.4,'Score cuts OR evaluate pilot gate'),(.1,'Seal choices; reset and train candidates')]:
  a.text(.5,y,text,ha='center',va='center',fontsize=PT_SMALL,transform=a.transAxes,bbox=dict(boxstyle='round,pad=.38',fc='#EDF3F1',ec='#B4C9BF',lw=.6))
 for y in [.84,.54,.24]:a.annotate('',(.5,y-.07),(.5,y+.04),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=.7,color=COLORS['point_mlp']))
 b=fig.add_subplot(gs[0,1]);clean(b,'x');title(b,'B','Two prespecified pooled comparisons')
 z=contrasts[contrasts.primary_comparison]
 for j,(_,r) in enumerate(z.iterrows()):
  b.errorbar(r.mean_improvement,j,xerr=[[r.mean_improvement-r.ci975_low],[r.ci975_high-r.mean_improvement]],fmt='o',ms=3.5,color=PC[r.baseline],capsize=2,lw=LW_DATA)
  b.text(.02,j+.21,PN[r.baseline],fontsize=PT_SMALL,transform=b.get_yaxis_transform())
 b.axvline(.01,ls=':',color='#929BA2',lw=LW_REF);b.set_yticks([]);b.set_ylim(-.45,1.6);b.set_xlabel('baseline − estimated NMSE',fontsize=PT_LABEL)
 c=fig.add_subplot(gs[1,0]);clean(c,'y');title(c,'C','Calibration size and label noise')
 for policy in POLICIES[:3]:
  for noise,ls in [(0.,'--'),(.5,'-')]:
   z=source[source.family.eq('all')&source.policy.eq(policy)&source.calibration_noise_sd.eq(noise)].sort_values('calibration_rows')
   c.plot(z.calibration_rows,z.mean_regret,color=PC[policy],ls=ls,marker='o',ms=2.5,lw=LW_DATA)
 c.set_xscale('log',base=2);c.set_xticks([64,256,1024],[64,256,1024]);c.set_xlabel('calibration labels',fontsize=PT_LABEL);c.set_ylabel('menu regret',fontsize=PT_LABEL)
 c.text(.48,.82,'solid: noise SD 0.5\ndashed: noiseless',ha='left',va='top',transform=c.transAxes,fontsize=PT_SMALL)
 d=fig.add_subplot(gs[1,1]);clean(d,'y');title(d,'D','Primary condition: family differences')
 fam=['matching','quartet','nested_prefix','random_interactions']
 for j,policy in enumerate(POLICIES[:3]):
  z=source[source.family.isin(fam)&source.policy.eq(policy)&source.calibration_rows.eq(256)&source.calibration_noise_sd.eq(.5)].set_index('family').loc[fam]
  d.errorbar(np.arange(4)+(j-1)*.12,z.mean_regret,yerr=[z.mean_regret-z.regret_ci_low,z.regret_ci_high-z.mean_regret],fmt='o',ms=2.7,lw=LW_DATA,color=PC[policy],capsize=1.3)
 d.set_xticks(range(4),['matching','quartic','nested','random']);d.set_ylabel('menu regret',fontsize=PT_LABEL)
 e=fig.add_subplot(gs[2,0]);clean(e,'y');title(e,'E','Secondary adaptive construction; ALS fits')
 for j,(policy,color,label) in enumerate([('estimated_adaptive_dp',COLORS['shunting'],'estimated tree'),('oracle_best_trained_menu',COLORS['point_mlp'],'best of twelve (oracle)'),('oracle_adaptive_dp',COLORS['oracle'],'true-target tree (oracle)')]):
  z=absolute[absolute.family.isin(fam)&absolute.policy.eq(policy)&absolute.calibration_rows.eq(256)&absolute.calibration_noise_sd.eq(.5)].set_index('family').loc[fam]
  e.errorbar(np.arange(4)+(j-1)*.12,z.mean_test_nmse,yerr=[z.mean_test_nmse-z.ci95_low,z.ci95_high-z.mean_test_nmse],fmt='o',ms=2.8,lw=LW_DATA,color=color,capsize=1.3,label=label)
 e.set_xticks(range(4),['matching','quartic','nested','random']);e.set_yscale('log');e.set_ylim(2e-4,3);e.set_ylabel('clean-test NMSE',fontsize=PT_LABEL)
 e.legend(frameon=False,fontsize=PT_SMALL,loc='center left',bbox_to_anchor=(.0,.5))
 f=fig.add_subplot(gs[2,1]);clean(f,'y');title(f,'F','Selection cost before final training')
 z=records[records.calibration_rows.eq(256)&records.calibration_noise_sd.eq(.5)].copy()
 for x,values,color in [(0,z.estimator_seconds+z.score_seconds,COLORS['shunting']),(1,z.pilot_seconds,COLORS['local'])]:
  q=np.quantile(values,[.25,.5,.75]);f.errorbar(x,q[1],yerr=[[q[1]-q[0]],[q[2]-q[1]]],fmt='o',color=color,ms=4,lw=LW_DATA,capsize=2)
 f.set_xticks([0,1],['estimate +\nscore twelve','two sweeps ×\ntwelve trees']);f.set_xlim(-.4,1.4);f.set_ylabel('elapsed seconds\n(median, IQR)',fontsize=PT_LABEL)
 if has_end:
  end_to_end_panels(fig,gs)
 handles=[Line2D([],[],color=PC[p],marker='o',ms=3,lw=LW_DATA,label=PN[p]) for p in POLICIES[:3]]
 fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.53,.995),frameon=False,fontsize=PT_SMALL)
 save(fig,41,'H' if has_end else 'F',[CALIB/x for x in ['policy_summary.csv','figure_absolute_error_summary.csv','primary_contrasts.csv','calibration_selection_records.csv']]+([end_path,CALIB/'end_to_end/contrasts.csv',CALIB/'end_to_end/endpoints.csv'] if has_end else []))

COND_RULES=['exact_path','calibrated_broadcast','broadcast_projection','subtree_projection']
COND_COLORS=[COLORS['shunting'],COLORS['local'],'#648C9B',COLORS['oracle']]
COND_NAMES=['exact path','fixed calibrated broadcast','one oracle profile','two subtree profiles']

def s42():
 source=COND/'summaries/fresh'
 summary=pd.read_csv(source/'learning_summary.csv')
 contrasts=pd.read_csv(source/'paired_contrasts.csv')
 paired=pd.read_csv(source/'paired_seed_contrasts.csv')
 geometry=pd.read_csv(source/'credit_geometry_summary.csv')
 bounds=pd.read_csv(COND/'interaction_bound/population_bounds.csv')
 fig=plt.figure(figsize=(FIG_W,6.25))
 gs=fig.add_gridspec(3,2,left=.10,right=.97,top=.88,bottom=.075,wspace=.45,hspace=.65)
 a=fig.add_subplot(gs[0,0]);a.axis('off');title(a,'A','Fixed physical shape; different grouping')
 xy={0:(.1,.15),1:(.33,.15),2:(.67,.15),3:(.9,.15),4:(.215,.57),5:(.785,.57),6:(.5,.93)}
 for c,p in [(0,4),(1,4),(2,5),(3,5),(4,6),(5,6)]:a.plot([xy[c][0],xy[p][0]],[xy[c][1],xy[p][1]],color='#53656D',lw=1)
 for n,(x,y) in xy.items():a.plot(x,y,'o',ms=6 if n==6 else 5,color=COLORS['shunting'] if n<2 or n==4 else COLORS['local'] if n<6 else '#53656D',zorder=3)
 for n in range(4):a.text(xy[n][0],-.01,str(n),ha='center',fontsize=PT_SMALL)
 a.text(.5,-.24,'4 E + 6 I contacts; 6 couplings; 16 conductances',ha='center',fontsize=PT_SMALL)
 a.text(.5,-.45,'Input groupings: 01 | 23, 02 | 13, 03 | 12',ha='center',fontsize=PT_SMALL)
 a.set_xlim(-.02,1.02);a.set_ylim(-.55,1.05)
 b=fig.add_subplot(gs[0,1]);clean(b,'y');title(b,'B','Interaction obstruction: population bound')
 key='additive_subtree_nmse_lower_bound'
 if key not in bounds: key=[x for x in bounds if 'nmse' in x and 'lower' in x][0]
 # Zero-bound rows are compatible; each seed averages its six incompatible pairs.
 z=bounds[bounds[key]>1e-15].groupby('seed')[key].mean().sort_index()
 stat=json.loads((COND/'interaction_bound/seed_summary.json').read_text())
 b.scatter(np.linspace(-.13,.13,len(z)),z,s=9,color=COLORS['local'],alpha=.5)
 from build_morphology_credit_figure_tables import bootstrap
 mean,lo,hi=bootstrap(z.to_numpy())
 b.errorbar(0,mean,yerr=[[mean-lo],[hi-mean]],fmt='D',ms=4,color=COLORS['local'],lw=LW_DATA,capsize=2)
 b.plot(1,0,'o',color=COLORS['shunting'],ms=4);b.set_xticks([0,1],['incompatible','compatible']);b.set_xlim(-.4,1.4);b.set_ylim(-.0007,max(z)*1.23);b.set_ylabel('population NMSE\nlower bound',fontsize=PT_LABEL)
 b.text(.03,.97,'Converged quadrature; not observed loss',va='top',transform=b.transAxes,fontsize=PT_SMALL)
 step=summary.step.max()
 for j,opt in enumerate(['adam','sgd']):
  ax=fig.add_subplot(gs[1,j]);clean(ax,'y');title(ax,'CD'[j],opt.upper()+': all credit rules')
  for x,rule in enumerate(COND_RULES):
   for compatible,offset,marker in [(True,-.1,'o'),(False,.1,'s')]:
    r=summary[summary.optimizer.eq(opt)&summary.credit_rule.eq(rule)&summary.compatible.eq(compatible)&summary.step.eq(step)].iloc[0]
    ax.errorbar(x+offset,r.mean_test_nmse,yerr=[[r.mean_test_nmse-r.ci95_low],[r.ci95_high-r.mean_test_nmse]],fmt=marker,ms=3,color=COND_COLORS[x],mfc=COND_COLORS[x] if compatible else 'white',lw=LW_DATA,capsize=1.5)
  ax.set_xticks(range(4),['exact','fixed','one','two']);ax.set_yscale('log');ax.set_ylim(5e-5,.12);ax.set_ylabel('test NMSE',fontsize=PT_LABEL)
  if j==0:ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#53656D',ms=3,label='compatible'),Line2D([],[],marker='s',ls='',color='#53656D',mfc='white',ms=3,label='incompatible')],frameon=False,fontsize=PT_SMALL,loc='center',bbox_to_anchor=(.5,1.0),ncol=2)
 e=fig.add_subplot(gs[2,0]);clean(e,'y');title(e,'E','Compatible trees: broadcast credit effect')
 for j,opt in enumerate(['adam','sgd']):
  key='calibrated_broadcast_minus_exact_path__compatible_True'
  r=contrasts[contrasts.optimizer.eq(opt)&contrasts.contrast.eq(key)].iloc[0]
  zz=paired[paired.optimizer.eq(opt)&paired.contrast.eq(key)]
  value='difference' if 'difference' in zz else 'nmse_difference'
  e.scatter(j+np.linspace(-.12,.12,len(zz)),zz[value],s=8,color=COND_COLORS[j],alpha=.35)
  e.errorbar(j,r.mean_difference,yerr=[[r.mean_difference-r.ci95_low],[r.ci95_high-r.mean_difference]],fmt='D',ms=3.5,color=COND_COLORS[j],lw=LW_DATA,capsize=2)
 e.axhline(0,color='#A0A9AF',lw=LW_REF);e.set_xticks([0,1],['Adam','SGD']);e.set_xlim(-.4,1.4);e.set_ylabel('fixed broadcast − exact\nNMSE',fontsize=PT_LABEL)
 f=fig.add_subplot(gs[2,1]);clean(f,'y');title(f,'F','Adam: common-state gradient alignment')
 for rule,color in zip(COND_RULES,COND_COLORS):
  z=geometry[geometry.optimizer.eq('adam')&geometry.credit_rule.eq(rule)&geometry.compatible].sort_values('step')
  f.plot(z.step,z.gradient_cosine,marker='o',ms=2.5,color=color,lw=LW_DATA)
 f.set_xscale('symlog',linthresh=10);f.set_xticks([0,10,100,1000],[0,10,100,1000]);f.set_ylim(min(.6,float(geometry[geometry.optimizer.eq('adam')&geometry.compatible].gradient_cosine.min())-.04),1.02);f.set_ylabel('calibration-gradient\ncosine',fontsize=PT_LABEL);f.set_xlabel('training update',fontsize=PT_LABEL)
 handles=[Line2D([],[],color=c,marker='o',ms=3,lw=LW_DATA,label=n) for c,n in zip(COND_COLORS,COND_NAMES)]
 fig.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.53,.995),frameon=False,fontsize=PT_SMALL)
 save(fig,42,'F',[source/x for x in ['learning_summary.csv','paired_contrasts.csv','paired_seed_contrasts.csv','credit_geometry_summary.csv']]+[COND/'interaction_bound/population_bounds.csv',COND/'interaction_bound/seed_summary.json'])

def main():
 apply_neurips_style();s40();s41();s42()

if __name__=='__main__':main()
