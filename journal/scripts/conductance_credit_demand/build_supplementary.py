#!/usr/bin/env python3
"""Native supplementary controls; every curve and interval has a source row."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter
HERE=Path(__file__).resolve().parent;J=HERE.parents[1];ROOT=J/'source_data/conductance_credit_demand';sys.path.insert(0,str(HERE.parent))
from figure_canvas import NativeCanvas,Margins,COLORS,PT_ANNOT,PT_LEGEND,LW_DATA,LW_REF
from report import interval
COLOR=dict(exact=COLORS['bp'],unit_broadcast=COLORS['scalar'],calibrated_broadcast=COLORS['additive'],ancestry_three_oracle=COLORS['oracle'])
LABEL=dict(exact='Exact path',unit_broadcast='Unit broadcast',calibrated_broadcast='Initial profile',ancestry_three_oracle='Three profiles (oracle)')
STYLE=dict(exact='-',unit_broadcast='--',calibrated_broadcast=':',ancestry_three_oracle='-.')

def clean_curves(path):
 d=pd.read_csv(path);d=d[d.phase.isin(['fresh','extension'])]
 # The continuation records its unchanged4096 start; one point per actual fit.
 d=d.sort_values('phase').drop_duplicates(['seed','task','optimizer','rule','rate','step']);return d

def curve(ax,d,task,optimizer,rules,table,panel):
 for rule in rules:
  part=d[(d.task==task)&(d.optimizer==optimizer)&(d.rule==rule)&d.selected_rate];rows=[]
  for step,g in part.groupby('step'):
   mean,lo,hi=interval(g.test_nmse);rows.append(dict(panel=panel,task=task,optimizer=optimizer,rule=rule,step=step,mean=mean,ci_low=lo,ci_high=hi,n=len(g)))
  r=pd.DataFrame(rows);table+=rows;ax.plot(r.step,r['mean'],color=COLOR[rule],ls=STYLE[rule],lw=LW_DATA,label=LABEL[rule]);ax.fill_between(r.step,r.ci_low,r.ci_high,color=COLOR[rule],alpha=.10,lw=0)
 ax.axvline(4096,color=COLORS['mute'],lw=LW_REF,ls=':');ax.set_yscale('log');ax.set_xlim(0,16800);ax.set_ylim(1e-8,1.6);ax.set_xticks([0,4096,16384],['0','4,096','16,384']);ax.set_xlabel('Training updates');ax.set_ylabel('Test NMSE');ax.yaxis.set_major_locator(FixedLocator([1e-8,1e-6,1e-4,1e-2,1]));ax.yaxis.set_major_formatter(FixedFormatter(['10⁻⁸','10⁻⁶','10⁻⁴','10⁻²','1']));ax.minorticks_off()

def first():
 dest=ROOT/'figures';dest.mkdir(exist_ok=True);table=[];d=clean_curves(ROOT/'summaries/all_curves.csv');cv=NativeCanvas(425/72,2,row_weights=[165,165],hgutter_pt=28,vgutter_pt=55,margins=Margins(left=63,right=18,top=24,bottom=42));a=cv.panel('A',0,0,6,title='Ungated, independent inputs',grid='y');b=cv.panel('B',0,6,6,title='Gated, conflicting inputs',grid='y')
 for ax,task,pan in [(a,'ungated_independent','A'),(b,'gated_conflict','B')]:curve(ax,d,task,'adam',['exact','unit_broadcast','calibrated_broadcast'],table,pan)
 a.legend(frameon=False,fontsize=PT_LEGEND,loc='upper right',handlelength=1.7)
 c=cv.panel('C',1,0,6,title='A small precision benefit remains',grid='x');contr=pd.read_csv(ROOT/'summaries/paired_contrasts.csv');seeds=pd.read_csv(ROOT/'summaries/paired_seed_contrasts.csv')
 for y,opt in [(1,'adam'),(0,'sgd')]:
  mask=(contr.phase=='extension')&(contr.optimizer==opt)&(contr.rate_scope=='selected')&(contr.rule=='calibrated_broadcast')&(contr.contrast=='target_gap');r=contr[mask].iloc[0];p=seeds[(seeds.phase=='extension')&(seeds.optimizer==opt)&(seeds.rate_scope=='selected')&(seeds.rule=='calibrated_broadcast')&(seeds.contrast=='target_gap')];c.scatter(p.value*1000,y+np.linspace(-.13,.13,len(p)),s=8,color=COLORS['mute'],alpha=.4,linewidths=0);c.errorbar(r['mean']*1000,y,xerr=[[1000*(r['mean']-r.ci_low)],[1000*(r.ci_high-r['mean'])]],fmt='o',color=COLORS['bp'],ms=4,lw=LW_DATA,capsize=2);table.append(dict(panel='C',**r.to_dict()))
 c.axvline(0,color=COLORS['mute'],ls=':',lw=LW_REF);c.set_yticks([1,0],['Adam','SGD']);c.set_ylim(-.5,1.5);c.set_xlabel('Gated-task NMSE gap (×10⁻³)');c.text(.98,.98,'Initial profile − exact\n16,384-update window',transform=c.transAxes,ha='right',va='top',fontsize=PT_ANNOT)
 e=cv.panel('D',1,6,6,title='Eligible parameters retain most credit',grid='y');g=pd.read_csv(ROOT/'summaries/all_endpoints.csv');g=g[(g.phase=='extension')&(g.optimizer=='adam')&(g.rule=='exact')&g.selected_rate];
 for offset,metric,col,label in [(-.12,'path_rank_one_capture',COLORS['bp'],'Best rank-one path capture'),(.12,'eligibility_calibrated_oracle_capture',COLORS['additive'],'Eligibility-weighted profile capture')]:
  for i,task in enumerate(['ungated_independent','gated_conflict']):
   v=g[g.task==task][metric];mean,lo,hi=interval(v);e.errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=col,ms=4,lw=LW_DATA,capsize=2,label=label if i==0 else None);table.append(dict(panel='D',task=task,metric=metric,mean=mean,ci_low=lo,ci_high=hi))
 e.set_xticks([0,1],['Ungated','Gated conflict']);e.set_xlim(-.45,1.45);e.set_ylim(.8,1.02);e.set_yticks([.8,.85,.9,.95,1.]);e.set_ylabel('Captured squared energy');e.legend(frameon=False,fontsize=PT_LEGEND,loc='lower left',handlelength=1.0)
 pd.DataFrame(table).to_csv(dest/'supplement_first_conductance_source.csv',index=False);cv.save(dest/'supplement_first_conductance.pdf')

def second():
 root=ROOT/'opponent';dest=root/'supplementary_figures';dest.mkdir(exist_ok=True);table=[];d=clean_curves(root/'summaries/all_curves.csv');cv=NativeCanvas(430/72,2,row_weights=[165,165],hgutter_pt=28,vgutter_pt=58,margins=Margins(left=65,right=18,top=25,bottom=40));a=cv.panel('A',0,0,6,title='SGD also favors routed credit',grid='y');curve(a,d,'opposed_strong','sgd',['exact','unit_broadcast','calibrated_broadcast','ancestry_three_oracle'],table,'A');a.set_ylim(1e-5,1.6);a.legend(frameon=False,fontsize=PT_LEGEND,loc='center right',handlelength=1.7)
 b=cv.panel('B',0,6,6,title='Wider conductance bounds retain the gap',grid='x');contr=pd.read_csv(root/'bound_sensitivity/summaries/paired_contrasts.csv')
 for y,opt,bound in [(3,'adam',7),(2,'adam',20),(1,'sgd',7),(0,'sgd',20)]:
  r=contr[(contr.optimizer==opt)&(contr.bound==bound)&(contr.rule=='calibrated_broadcast')&(contr.contrast=='task_difference_in_gap')].iloc[0];b.errorbar(r['mean'],y,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='o' if bound==7 else 's',mfc='white' if bound==7 else COLORS['bp'],color=COLORS['bp'],ms=4,lw=LW_DATA,capsize=2);table.append(dict(panel='B',**r.to_dict()))
 b.set_yticks([3,2,1,0],['Adam ±7','Adam ±20','SGD ±7','SGD ±20']);b.set_ylim(-.6,3.6);b.set_xlim(0,.95);b.axvline(0,color=COLORS['mute'],ls=':',lw=LW_REF);b.set_xlabel('Opposed − aligned credit deficit (NMSE)');b.text(.98,.98,'Log-conductance bounds',transform=b.transAxes,ha='right',va='top',fontsize=PT_ANNOT)
 c=cv.panel('C',1,0,6,title='The gap precedes parameter-bound contact',grid='y');g=pd.read_csv(root/'bound_sensitivity/summaries/precontact_gaps.csv');g=g[(g.optimizer=='adam')&(g.rule=='calibrated_broadcast')];c.scatter(g.last_precontact_checkpoint,g.gap,color=COLORS['additive'],s=15,alpha=.75,linewidths=0);c.axhline(0,color=COLORS['mute'],ls=':',lw=LW_REF);c.set_xscale('symlog',linthresh=256);c.set_xticks([256,1024,4096,8192],['256','1,024','4,096','8,192']);c.set_xlim(200,10000);c.set_ylim(-.035,1.12);c.set_xlabel('Last checkpoint before first bound contact');c.set_ylabel('Initial profile − exact NMSE');c.text(.03,.97,'All 20 gaps positive',transform=c.transAxes,ha='left',va='top',fontsize=PT_ANNOT)
 for r in g.to_dict('records'):table.append(dict(panel='C',**r))
 e=cv.panel('D',1,6,6,title='Every development regime is retained',grid='y');dev=pd.read_csv(root/'development_rate_selection.csv');dev=dev[dev.optimizer=='adam'];tasks=['aligned_strong','opposed_strong','opposed_moderate','opposed_ungated']
 for offset,rule in zip([-.24,-.08,.08,.24],['exact','ancestry_three_oracle','calibrated_broadcast','unit_broadcast']):
  vals=[]
  for i,task in enumerate(tasks):
   p=dev[(dev.task==task)&(dev.rule==rule)];r=p.loc[p.validation_nmse.idxmin()];vals.append(r.validation_nmse);table.append(dict(panel='D',**r.to_dict()))
  e.plot(np.arange(4)+offset,vals,ls='none',marker='o',ms=3.6,color=COLOR[rule],label=LABEL[rule])
 e.set_yscale('log');e.set_ylim(1e-5,1.3);e.set_xticks(range(4),['Aligned\nstrong','Opposed\nstrong','Opposed\nmoderate','Opposed\nungated']);e.set_ylabel('Development validation NMSE');e.set_xlim(-.5,3.5);e.minorticks_off()
 pd.DataFrame(table).to_csv(dest/'supplement_opponent_controls_source.csv',index=False);cv.save(dest/'supplement_opponent_controls.pdf')

if __name__=='__main__':first();second()
