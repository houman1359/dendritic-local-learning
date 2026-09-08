#!/usr/bin/env python3
"""Five-panel main-ready depth figure; original source and builder untouched."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import analyze
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas,Margins,COLORS,PT_LABEL,PT_ANNOT,PT_LEGEND,LW_DATA,LW_REF,LW_EDGE

SPECS={('exact_autograd_bp_recipe',1):('D1: exact BP',COLORS['point_mlp'],'--'),('exact_autograd_bp_recipe',3):('D3: exact BP',COLORS['bp'],'-'),('path_transport',3):('D3: exact LocalCA',COLORS['pathway'],'-'),('per_soma_shared',3):('D3: shared LocalCA',COLORS['local'],'-'),('broadcast_autograd_bp_recipe',3):('D3: broadcast; BP recipe',COLORS['additive'],'-'),('broadcast_autograd_localca_recipe',3):('D3: broadcast; LocalCA recipe',COLORS['local'],':')}

def schematic(a):
 a.set_xlim(0,1);a.set_ylim(-.17,1.03)
 a.text(.012,.87,'Nested gain task',fontsize=PT_LABEL,fontweight='bold',va='center')
 a.text(.012,.64,'Distal class signal × nuisance gains',fontsize=PT_ANNOT,va='center')
 a.text(.012,.37,'Separate inhibitory sensors:',fontsize=PT_ANNOT,va='center')
 tiers=[('Fine',COLORS['additive']),('Coarse',COLORS['pathway']),('Global',COLORS['local'])]
 for x,(name,color) in zip([.052,.17,.295],tiers):
  a.scatter([x],[.15],s=29,color=color,zorder=3)
  a.text(x+.02,.15,name,fontsize=PT_ANNOT,va='center')
 a.text(.012,-.095,'Gain groups: 4 fine / 2 coarse / 1 global',fontsize=PT_ANNOT,va='center')
 def node(x,y,tier):
  a.scatter([x],[y],s=35,color=tiers[tier][1],edgecolors='white',linewidths=.55,zorder=3)
 def edge(x,y,xx,yy):a.plot([x,xx],[y,yy],color=COLORS['edge'],lw=LW_EDGE,zorder=1)
 def soma(x,y):a.scatter([x],[y],s=48,color=COLORS['ink'],zorder=3)
 a.text(.59,.91,'D1 [8]',ha='center',fontsize=PT_LABEL)
 xs=np.linspace(.445,.735,8)
 for k,x in enumerate(xs):edge(x,.65,.59,.13);node(x,.65,0 if k<4 else 1 if k<6 else 2)
 soma(.59,.13);a.text(.59,-.095,'One physical stage',ha='center',fontsize=PT_ANNOT)
 a.text(.88,.91,'D3 [2,1,2]',ha='center',fontsize=PT_LABEL)
 soma(.88,.03)
 for proximal,mid,leaves in [(.82,.82,[.785,.855]),(.955,.955,[.92,.99])]:
  edge(.88,.03,proximal,.25);node(proximal,.25,2)
  edge(proximal,.25,mid,.46);node(mid,.46,1)
  for leaf in leaves:edge(mid,.46,leaf,.69);node(leaf,.69,0)
 a.text(.88,-.095,'Three physical stages',ha='center',fontsize=PT_ANNOT)


def plot_condition(ax,frame,arm,depth,metric,table,panel,show_ci=True):
 p=frame[(frame.arm==arm)&(frame.depth==depth)&(frame.metric==metric)].sort_values('epoch');label,color,ls=SPECS[(arm,depth)];scale=100 if metric=='test_accuracy' else 1
 if show_ci:ax.fill_between(p.epoch,scale*p.ci95_low,scale*p.ci95_high,color=color,alpha=.11,lw=0)
 line,=ax.plot(p.epoch,scale*p['mean'],color=color,ls=ls,lw=LW_DATA,label=label)
 table.extend({'panel':panel,**r} for r in p.to_dict('records'))
 return line


def main():
 out=analyze.OUT;valid=json.loads((out/'analysis_validation.json').read_text());assert valid['status']=='passed'
 s=pd.read_csv(out/'condition_trajectory_summary.csv',float_precision='round_trip');g=pd.read_csv(out/'paired_trajectory_summary.csv',float_precision='round_trip');table=[]
 c=NativeCanvas(490/72,3,row_weights=[128,130,130],hgutter_pt=35,vgutter_pt=42,margins=Margins(left=52,right=18,top=27,bottom=36))
 a=c.panel('A',0,0,12,schematic=True,title='Same resource budget; different serial organization',lock=False);schematic(a)
 b=c.panel('B',1,0,6,title='Serial depth retains its forward benefit',grid='y')
 for arm,depth in [('exact_autograd_bp_recipe',1),('exact_autograd_bp_recipe',3),('path_transport',3),('per_soma_shared',3)]:plot_condition(b,s,arm,depth,'test_accuracy',table,'B')
 b.axvline(180,color=COLORS['mute'],ls=':',lw=LW_REF);b.set_xlim(0,610);b.set_ylim(32,102);b.set_xticks([0,180,400,600]);b.set_xlabel('Epoch');b.set_ylabel('Test accuracy (%)');b.legend(loc='lower right',frameon=False,fontsize=PT_LEGEND,handlelength=1.5,labelspacing=0.0)
 b.text(.03,.98,'10 seeds per condition',transform=b.transAxes,va='top',fontsize=PT_ANNOT)
 ax=c.panel('C',1,6,6,title='Accuracy ranking changes with budget',grid='y')
 p=g[g.metric=='test_accuracy'].sort_values('epoch');ax.fill_between(p.epoch,p.ci95_low,p.ci95_high,color=COLORS['pathway'],alpha=.14,lw=0);ax.plot(p.epoch,p['mean'],color=COLORS['pathway'],lw=LW_DATA);ax.axhline(0,color=COLORS['edge'],lw=LW_REF,ls='--');ax.axvline(180,color=COLORS['mute'],lw=LW_REF,ls=':')
 for epoch,text,offset in [(180,'+10.86',(12,0)),(486,'−2.94',(-29,-5)),(600,'−1.52',(-26,17))]:
  r=p[p.epoch==epoch].iloc[0];ax.scatter(epoch,r['mean'],s=21,color=COLORS['pathway'],zorder=4);ax.annotate(text,(epoch,r['mean']),xytext=offset,textcoords='offset points',fontsize=PT_ANNOT,va='center',ha='left')
 ax.set_xlim(0,615);ax.set_ylim(min(-5.0,float(p.ci95_low.min())-.2),max(14.,float(p.ci95_high.max())+.5));ax.set_xticks([0,180,400,600]);ax.set_xlabel('Epoch');ax.set_ylabel('Exact − shared accuracy (pp)');table.extend({'panel':'C',**r} for r in p.to_dict('records'))
 d=c.panel('D',2,0,6,title='Validation loss by training recipe',grid='y');lines={}
 for arm,depth in [('exact_autograd_bp_recipe',1),('exact_autograd_bp_recipe',3),('broadcast_autograd_bp_recipe',3),('path_transport',3),('broadcast_autograd_localca_recipe',3),('per_soma_shared',3)]:lines[(arm,depth)]=plot_condition(d,s,arm,depth,'best_validation_loss',table,'D',show_ci=False)
 d.axvline(180,ymin=.44,color=COLORS['mute'],ls=':',lw=LW_REF);d.set_xlim(0,610);d.set_ylim(.07,.71);d.set_xticks([0,180,400,600]);d.set_xlabel('Epoch');d.set_ylabel('Best validation loss')
 handles=[lines[('broadcast_autograd_bp_recipe',3)],(lines[('per_soma_shared',3)],lines[('broadcast_autograd_localca_recipe',3)])]
 labels=['Broadcast; BP recipe','Shared / matched broadcast']
 d.legend(handles,labels,handler_map={tuple:HandlerTuple(ndivide=None)},loc='lower left',frameon=False,fontsize=PT_LEGEND,handlelength=1.4,labelspacing=0)
 e=c.panel('E',2,6,6,title='Loss and accuracy rank the rules differently',grid='y');p=g[g.metric=='test_cross_entropy'].sort_values('epoch')
 e.fill_between(p.epoch,p.ci95_low,p.ci95_high,color=COLORS['pathway'],alpha=.14,lw=0);e.plot(p.epoch,p['mean'],color=COLORS['pathway'],lw=LW_DATA);e.axhline(0,color=COLORS['edge'],lw=LW_REF,ls='--');e.axvline(180,color=COLORS['mute'],lw=LW_REF,ls=':')
 for epoch in [180,600]:r=p[p.epoch==epoch].iloc[0];e.scatter(epoch,r['mean'],s=21,color=COLORS['pathway'],zorder=4)
 e.set_xlim(0,615);e.set_ylim(min(-.085,float(p.ci95_low.min())-.005),max(.02,float(p.ci95_high.max())+.004));e.set_xticks([0,180,400,600]);e.set_xlabel('Epoch');e.set_ylabel('Exact − shared cross-entropy');e.text(.03,.04,'Negative: lower loss with exact credit',transform=e.transAxes,fontsize=PT_ANNOT,va='bottom');table.extend({'panel':'E',**r} for r in p.to_dict('records'))
 locks=c.lock_reserves();left=max(locks[name][0] for name in 'BCDE');right=max(locks[name][1] for name in 'BCDE')
 for name in 'BCDE':c.declare_reserve(name,left=left,right=right)
 dest=out/'figures';dest.mkdir(exist_ok=True);path=dest/'physical_depth_followup.pdf';c.save(path,name='physical_depth_followup',dpi=200);plt.close(c.fig);pd.DataFrame(table).to_csv(dest/'figure_source.csv',index=False)
 panelmap={'A':'Task/input routing schematic:gain groups4/2/1,modules4/2/2;D1[8]versusD3[2,1,2],64somata,eightnonsomatic compartments per soma,matched contacts and trainable resources. Drawings abbreviate contact fields,not exact analytic cancellation.','B':'Validation-selected testaccuracy trajectories,D1exactBP andD3exactBP/exactLocalCA/sharedLocalCA,with pointwise95percent whole-seed intervals.','C':'Paired D3exactLocalCA−sharedLocalCA testaccuracy gap in percentage points,pointwise95percent intervals;180/486/600 means marked.','D':'Allsix condition means of bestvalidationloss;stoppedD1runs carried forward;sharedLocalCA andmatched-broadcastLocalCA drawn separately with jointlegend.','E':'Paired D3exactLocalCA−sharedLocalCA testcross-entropy gap,pointwise95percent intervals;negative means exact lower loss.'}
 inputs=[out/'condition_trajectory_summary.csv',out/'paired_trajectory_summary.csv',out/'analysis_validation.json',analyze.J/'source_data/physical_depth_budget/canonical/extension_protocol.json']
 analyze.write(dest/'figure_provenance.json',{'figure_sha256':analyze.sha(path),'builder_sha256':analyze.sha(__file__),'sources_sha256':{str(p.relative_to(analyze.J)):analyze.sha(p) for p in inputs},'panel_map':panelmap,'all_original_outcomes_preserved':True,'new_training_runs':0,'scope':'Post-review trajectory analysis;10paired seeds,all50D3fitsreach600 and8/10D1stopbefore600. CIs pointwise descriptive,not simultaneous. Original main6 PDF retained in this directory parent; original180plot remainsS31.'})
 caption='''**Serial computation remains useful while the credit comparison changes with budget and metric. A,** Nested gains multiply the distal class signal. Separate inhibitory sensors report fine, coarse and global fields with 4/2/1 gain groups. D1 [8] and D3 [2,1,2] distribute eight compartments per soma across one or three stages; colors denote the 4/2/2 modules assigned fine/coarse/global inputs. Every network has 64 somata and matched contacts and trainable resources. The drawing indicates input access, not exact cancellation. **B,** Test accuracy of validation-selected states for D1 exact backpropagation (BP) and three D3 rules. Stopped runs retain their selected state. **C,** Paired D3 exact-path-minus-shared-soma LocalCA accuracy gap. The mean first crosses zero at 312 epochs and remains negative from 315 through 600; all ten pairs remain negative from 347. Marked means are +10.86, −2.94 and −1.52 percentage points at 180, 486 and 600 epochs. **D,** Best validation loss for all six conditions, with colors as in B. Broadcast autograd with the LocalCA recipe nearly overlaps shared-soma LocalCA; both curves and a joint legend are retained. All 50 D3 fits reach 600 epochs with declining late validation loss; eight of ten D1 references stop earlier. **E,** Paired test cross-entropy gap at the same validation-selected states. Exact LocalCA retains lower mean loss throughout epochs 180–600 despite lower classification accuracy after the crossing. Shading in B, C and E shows pointwise descriptive 95% whole-seed bootstrap intervals from 10,000 draws of ten paired seeds. Vertical dotted lines mark 180 epochs. Both recipes use Adam with their original learning rates and parameter groups. These post-review summaries preserve the original fits; the reversal is budget-indexed and neither rule is established as the converged winner.'''
 (out/'FIGURE_CAPTION.md').write_text(caption+'\n');print(path)

if __name__=='__main__':main()
