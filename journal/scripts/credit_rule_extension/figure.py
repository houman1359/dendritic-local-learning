#!/usr/bin/env python3
"""Budget-indexed, shared-style visualization of the balanced extension."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,FixedFormatter
import run
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas,Margins,COLORS,PT_LEGEND,PT_ANNOT,LW_DATA,LW_REF

RULES=['exact','unit_broadcast','calibrated_broadcast','sign_broadcast']
COLOR={'exact':COLORS['bp'],'unit_broadcast':COLORS['scalar'],'calibrated_broadcast':COLORS['additive'],'sign_broadcast':COLORS['ink']}
LABEL={'exact':'Exact path','unit_broadcast':'Unit broadcast','calibrated_broadcast':'Initial profile','sign_broadcast':'Initial sign'}
STYLE={'exact':'-','unit_broadcast':'-','calibrated_broadcast':'-','sign_broadcast':':'}

def curve(ax,d,table,panel,task,view,optimizer='adam',legend=False,legend_loc='upper right'):
 upper=2.0
 for rule in RULES:
  part=d[(d.task==task)&(d.optimizer==optimizer)&d[view]&(d.rule==rule)&(d.step>=64)]
  w=part.pivot(index='seed',columns='step',values='test_nmse').sort_index();v=w.to_numpy(float)
  ix=np.random.default_rng(210999).integers(20,size=(10000,20));boot=v[ix].mean(axis=1)
  lo,hi=np.quantile(boot,[.025,.975],axis=0);mean=v.mean(axis=0);steps=w.columns.to_numpy(int);upper=max(upper,float(hi.max())*1.08)
  ax.fill_between(steps,lo,hi,color=COLOR[rule],alpha=.1,lw=0)
  ax.plot(steps,mean,color=COLOR[rule],ls=STYLE[rule],lw=LW_DATA,label=LABEL[rule],zorder=4 if rule=='sign_broadcast' else 3)
  for step,m,l,h in zip(steps,mean,lo,hi):table.append({'panel':panel,'task':task,'rule':rule,'rate_view':view,'optimizer':optimizer,'step':int(step),'mean':float(m),'ci95_low':float(l),'ci95_high':float(h)})
 floor=.045 if task=='quartet' else .0225
 ax.axhline(floor,color=COLORS['mute'],lw=LW_REF,ls=':');ax.axvline(1024,color=COLORS['edge'],lw=LW_REF,ls='--')
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlim(64,18000);ax.set_ylim(.018,upper)
 ax.set_xticks([64,1024,16384],['64','1,024','16,384']);ticks=[.02,.1,1]+[10**k for k in range(1,5) if 10**k<upper];ax.yaxis.set_major_locator(FixedLocator(ticks));ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
 ax.minorticks_off();ax.set_xlabel('Training updates');ax.set_ylabel('Test NMSE')
 if legend:ax.legend(frameon=False,fontsize=PT_LEGEND,loc=legend_loc,handlelength=1.65,labelspacing=.2)


def main():
 run.check_freeze();src=run.OUT/'summaries';dest=run.OUT/'figures';dest.mkdir(exist_ok=True)
 d=pd.read_csv(src/'all_curves.csv',float_precision='round_trip');contrasts=pd.read_csv(src/'paired_contrasts.csv',float_precision='round_trip');outcomes=pd.read_csv(src/'all_budget_outcomes.csv',float_precision='round_trip')
 c=NativeCanvas(490/72,3,row_weights=[125,125,132],hgutter_pt=35,vgutter_pt=53,margins=Margins(left=53,right=17,top=29,bottom=36));table=[]
 for letter,row,col,task,view,title in [('A',0,0,'matching','selected_rate','Pairwise: selected rates'),('B',0,6,'matching','common_rate','Pairwise: common rate'),('C',1,0,'quartet','selected_rate','Quartic: selected rates'),('D',1,6,'quartet','common_rate','Quartic: common rate')]:
  ax=c.panel(letter,row,col,6,title=title,grid='y');curve(ax,d,table,letter,task,view,legend=letter=='A')
 e=c.panel('E',2,0,6,title='Task-by-credit contrast over budget',grid='y')
 colors={'selected_rate':COLORS['bp'],'common_rate':COLORS['additive']}
 for view in ['selected_rate','common_rate']:
  for endpoint,ls,marker in [('terminal','-','o'),('validation_selected','--','s')]:
   part=contrasts[(contrasts.task=='quartet_minus_matching')&(contrasts.optimizer=='adam')&(contrasts.metric=='test_nmse')&(contrasts.contrast=='calibrated_broadcast minus exact interaction')&(contrasts.rate_view==view)&(contrasts.endpoint==endpoint)].sort_values('budget')
   e.plot(part.budget,part['mean'],color=colors[view],ls=ls,marker=marker,ms=3,lw=LW_DATA,label=('Selected' if view=='selected_rate' else 'Common')+'; '+('last' if endpoint=='terminal' else 'validation'))
   e.fill_between(part.budget,part.ci95_low,part.ci95_high,color=colors[view],alpha=.07,lw=0)
   table.extend({'panel':'E',**r} for r in part.to_dict('records'))
 e.axhline(0,color=COLORS['edge'],lw=LW_REF,ls=':');e.set_xscale('log',base=2);e.set_xlim(900,18500);e.set_xticks([1024,4096,16384],['1,024','4,096','16,384']);e.minorticks_off();e.set_xlabel('Maximum training updates');e.set_ylabel('Quartic − pairwise credit deficit');e.legend(frameon=False,fontsize=PT_LEGEND,loc='lower left',handlelength=1.6,labelspacing=.22)
 f=c.panel('F',2,6,6,title='Every quartic seed is retained',grid='both')
 for rule in ['exact','calibrated_broadcast']:
  part=outcomes[(outcomes.task=='quartet')&(outcomes.optimizer=='adam')&outcomes.selected_rate&(outcomes.rule==rule)&(outcomes.endpoint=='terminal')]
  w=part.pivot(index='seed',columns='budget',values='test_nmse')
  f.scatter(w[1024],w[16384],s=15,alpha=.8,color=COLOR[rule],label=LABEL[rule],linewidths=.25,edgecolors='white')
  table.extend({'panel':'F','seed':int(seed),'rule':rule,'test_nmse1024':float(r[1024]),'test_nmse16384':float(r[16384])} for seed,r in w.iterrows())
 f.plot([.025,1.5],[.025,1.5],lw=LW_REF,color=COLORS['edge'],ls='--');f.axhline(.045,lw=LW_REF,color=COLORS['mute'],ls=':');f.axvline(.045,ymax=.55,lw=LW_REF,color=COLORS['mute'],ls=':')
 f.set_xscale('log');f.set_yscale('log');f.set_xlim(.025,1.5);f.set_ylim(.025,1.5);f.set_xticks([.05,.2,1],['0.05','0.2','1']);f.set_yticks([.05,.2,1],['0.05','0.2','1']);f.minorticks_off();f.set_xlabel('Test NMSE at 1,024 updates');f.set_ylabel('Test NMSE at 16,384 updates');f.legend(frameon=False,fontsize=PT_LEGEND,loc='upper left',handlelength=1.4,labelspacing=.25)
 locks=c.lock_reserves();left=max(v[0] for v in locks.values());right=max(v[1] for v in locks.values())
 for name in c.axes:c.declare_reserve(name,left=left,right=right)
 path=dest/'credit_rule_extension.pdf';c.save(path,name='credit_rule_extension',dpi=200);plt.close(c.fig)
 pd.DataFrame(table).to_csv(dest/'figure_source.csv',index=False)
 # Companion controls retain the separate nested task, both SGD rate views, and bounds.
 c=NativeCanvas(375/72,2,row_weights=[135,135],hgutter_pt=35,vgutter_pt=53,margins=Margins(left=53,right=17,top=29,bottom=36));extra=[]
 for letter,row,col,task,view,opt,title in [('A',0,0,'nested','selected_rate','adam','Nested: selected Adam rates'),('B',0,6,'matching','common_rate','sgd','Pairwise: common SGD rate'),('C',1,0,'quartet','selected_rate','sgd','Quartic: selected SGD rates'),('D',1,6,'quartet','common_rate','sgd','Quartic: common SGD rate')]:
  ax=c.panel(letter,row,col,6,title=title,grid='y');curve(ax,d,extra,letter,task,view,opt,legend=letter=='A',legend_loc='center right')
 locks=c.lock_reserves();left=max(v[0] for v in locks.values());right=max(v[1] for v in locks.values())
 for name in c.axes:c.declare_reserve(name,left=left,right=right)
 path2=dest/'credit_rule_extension_controls.pdf';c.save(path2,name='credit_rule_extension_controls',dpi=200);plt.close(c.fig);pd.DataFrame(extra).to_csv(dest/'controls_figure_source.csv',index=False)
 inputs=[src/'all_curves.csv',src/'paired_contrasts.csv',src/'all_budget_outcomes.csv',run.OUT/'protocol_freeze.json']
 run.write(dest/'figure_provenance.json',{'figures_sha256':{p.name:run.sha(p) for p in [path,path2]},'source_sha256':{str(p.relative_to(run.J)):run.sha(p) for p in inputs},'builder_sha256':run.sha(__file__),'scope':'Existing observed seeds;20 paired blocks,720 extended trajectories; fixed historical rates and coefficient bounds. Bootstrap intervals descriptive. Dotted floor differs by task variance; curves are terminal states. Primary figure E also shows validation-selected endpoints.'})
 caption='''**Fixed profiles and exact credit under longer matched training budgets.** All 720 original algebraic trajectories were replayed and extended to 16,384 updates, retaining twenty seed blocks, three tasks, four rules, two optimizers and the union of the original selected/common rates. **A–D,** Mean test NMSE with pointwise descriptive 95% whole-seed bootstrap intervals for pairwise/quartic tasks under Adam, using the original selected rates or common rate 0.003. Initial-sign controls use dotted curves. Dashed vertical lines mark the historical 1,024-update cap; dotted horizontal lines mark noise-only NMSE 0.0225 (pairwise) or 0.045 (quartic). **E,** Quartic-minus-pairwise difference in calibrated-minus-exact NMSE at four maximum budgets. Solid/circle curves use terminal states; dashed/square curves use the minimum-validation-NMSE state among declared observations within each budget. Test results never select states. **F,** All twenty paired quartic endpoints under exact or calibrated-profile Adam credit at 1,024 and 16,384 updates; the diagonal indicates unchanged error. These are previously observed seed blocks with bounded coefficients and historical rates, not fresh confirmation or a convergence theorem.'''
 (run.OUT/'FIGURE_CAPTION.md').write_text(caption+'\n')
 (run.OUT/'CONTROLS_FIGURE_CAPTION.md').write_text('''**Nested-task and optimizer controls in the balanced extension. A,** Nested targets under the original selected Adam rates. **B,** Pairwise targets under the common SGD rate 0.03. **C,D,** Quartic targets under selected or common SGD rates. The selected SGD rates are 0.03 for exact/calibrated-profile credit and 0.01 for unit/sign broadcast. Curves retain all twenty original seed blocks, four rules and the same examples, initializations and parameter bounds. Lines are mean test NMSE; shading gives pointwise descriptive 95% paired whole-seed bootstrap intervals. Dotted horizontal lines mark noise-only NMSE, and dashed vertical lines mark 1,024 updates. The displayed histories are terminal states, with validation-selected outcomes supplied separately. The original rate choices were not reoptimized at the longer budget; unstable or poorly performing late outcomes remain visible. The common-SGD quartic axis expands to retain all curves and intervals.\n''')
 print(path);print(path2)

if __name__=='__main__':main()
