#!/usr/bin/env python3
"""Native main-figure candidate for the completed credit calibration bridge."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import run

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas, Margins, COLORS, PT_LABEL, PT_ANNOT, PT_SMALL, PT_LEGEND, LW_DATA, LW_REF, LW_EDGE
from journal_style import label_color

RULE_COLORS={'exact':COLORS['bp'],'unit_broadcast':COLORS['scalar'],'calibrated_broadcast':COLORS['additive']}
RULE_NAMES={'exact':'Exact path','unit_broadcast':'Unit broadcast','calibrated_broadcast':'Initial profile'}
TASK_NAMES={'matching':'Pairwise','quartet':'Quartic','nested':'Nested'}

def tree(ax,center,width,kind):
    xs=np.linspace(center-width/2,center+width/2,8)
    levels=[[(x,.20) for x in xs]]
    for level,y in enumerate([.40,.60,.80]):
        previous=levels[-1]; current=[]
        for i in range(0,len(previous),2):
            x=(previous[i][0]+previous[i+1][0])/2
            for child in previous[i:i+2]:
                ax.plot([x,child[0]],[y,child[1]],color=COLORS['edge'],lw=LW_EDGE,zorder=1)
            current.append((x,y))
            symbol='×' if level==0 or (level==1 and kind=='quartet') else '+'
            ax.text(x,y,symbol,ha='center',va='center',fontsize=PT_LABEL,
                bbox=dict(boxstyle='circle,pad=.12',fc='white',ec=COLORS['edge'],lw=LW_EDGE),zorder=3)
        levels.append(current)
    for i,x in enumerate(xs):
        ax.text(x,.10,f'$x_{i+1}$',ha='center',va='center',fontsize=PT_LABEL)
    ax.text(center,1.0,TASK_NAMES[kind]+' interactions',ha='center',va='center',fontsize=PT_LABEL)

def build():
    source=run.OUT/'summaries'; dest=run.OUT/'figures'; dest.mkdir(exist_ok=True)
    curves=pd.read_csv(source/'all_curves.csv')
    curves=curves[(curves.model=='algebraic')&(curves.optimizer=='adam')&curves.selected_rate]
    diagnostics=pd.read_csv(source/'all_diagnostics.csv')
    diag=diagnostics[(diagnostics.model=='algebraic')&(diagnostics.optimizer=='adam')&diagnostics.selected_rate&(diagnostics.rule=='exact')]
    contrasts=pd.read_csv(source/'paired_contrasts.csv')
    seeds=pd.read_csv(source/'paired_seed_contrasts.csv')
    canvas=NativeCanvas(490/72,3,row_weights=[132,125,123],hgutter_pt=37,vgutter_pt=47,
        margins=Margins(left=61,right=15,top=25,bottom=35))
    a=canvas.panel('A',0,0,12,schematic=True,title='One shared tree, two input-spectrum-matched targets',lock=False)
    a.set_xlim(0,1); a.set_ylim(-.26,1.12)
    tree(a,.235,.405,'matching'); tree(a,.765,.405,'quartet')
    a.text(.5,-.095,'Same eight inputs and tree; both targets exactly representable',ha='center',va='center',fontsize=PT_ANNOT)
    a.text(.5,-.235,'All eight input-sensitivity eigenvalues = 1/4; signed coefficients = ±0.5',ha='center',va='center',fontsize=PT_ANNOT)
    table=[]
    for letter,col,task in [('B',0,'matching'),('C',4,'quartet'),('D',8,'nested')]:
        title=TASK_NAMES[task]+(' (separate tree)' if task=='nested' else '')
        ax=canvas.panel(letter,1,col,4,title=title,grid='y')
        for rule in RULE_NAMES:
            part=curves[(curves.task==task)&(curves.rule==rule)]
            records=[]
            for step,group in part.groupby('step'):
                result=run.bootstrap(group.sort_values('seed').test_nmse)
                records.append(dict(step=step,**result)); table.append(dict(panel=letter,task=task,rule=rule,**records[-1]))
            frame=pd.DataFrame(records)
            ax.fill_between(frame.step,frame.ci95_low,frame.ci95_high,color=RULE_COLORS[rule],alpha=.13,lw=0)
            ax.plot(frame.step,frame['mean'],color=RULE_COLORS[rule],lw=LW_DATA,label=RULE_NAMES[rule])
            endpoint=part[part.step==1024]
            ax.scatter(np.full(len(endpoint),1024),endpoint.test_nmse,s=8,color=RULE_COLORS[rule],
                alpha=.65 if rule=='exact' else .38,linewidths=0,zorder=5 if rule=='exact' else 3)
        floor=.045 if task=='quartet' else .0225
        ax.axhline(floor,color=COLORS['mute'],lw=LW_REF,ls=':',zorder=0)
        ax.set_yscale('log');ax.set_ylim(.016,1.65);ax.set_xlim(0,1080)
        ax.set_xticks([0,512,1024],['0','512','1,024'])
        ax.yaxis.set_major_locator(FixedLocator([.02,.1,1.])); ax.yaxis.set_major_formatter(FixedFormatter(['0.02','0.1','1']))
        ax.minorticks_off();ax.set_xlabel('Training step')
        if col==0:
            ax.set_ylabel('Test NMSE')
            ax.legend(loc='upper right',frameon=False,fontsize=PT_LEGEND,handlelength=1.45,borderaxespad=.1,labelspacing=.35)
        else: ax.set_yticklabels([])
    e=canvas.panel('E',2,0,6,title='Quartic targets incur a larger deficit',grid='x')
    contrast='calibrated_broadcast minus exact interaction'
    for y,sensitivity,label in [(1,'selected_rate','Rule-specific rates'),(0,'common_rate','Common rate')]:
        row=contrasts[(contrasts.sensitivity==sensitivity)&(contrasts.task=='quartet_minus_matching')&(contrasts.optimizer=='adam')&(contrasts.contrast==contrast)].iloc[0]
        part=seeds[(seeds.sensitivity==sensitivity)&(seeds.task=='quartet_minus_matching')&(seeds.optimizer=='adam')&(seeds.contrast==contrast)].sort_values('seed')
        jitter=np.linspace(-.10,.10,len(part))
        e.scatter(part.difference,y+jitter,s=10,color=COLORS['mute'],alpha=.45,linewidths=0,zorder=2)
        e.errorbar(row['mean'],y,xerr=[[row['mean']-row.ci95_low],[row.ci95_high-row['mean']]],fmt='D',color=COLORS['bp'],ms=4.6,lw=LW_DATA,capsize=2,zorder=4)
        table.append(dict(panel='E',**row.to_dict()))
    e.axvline(0,color=COLORS['edge'],lw=LW_REF,ls='--');e.set_xlim(-.03,1.13);e.set_ylim(-.65,1.6)
    e.set_yticks([1,0],['Rule-specific\nrates','Common\nrate']);e.set_xlabel('Quartic − pairwise credit deficit')
    e.text(.99,.98,'20 paired seeds',transform=e.transAxes,ha='right',va='top',fontsize=PT_ANNOT)
    f=canvas.panel('F',2,6,6,title='Tasks differ in credit dimension',grid='y')
    for task,marker,style in [('matching','o','-'),('quartet','s','--'),('nested','^',':')]:
        values=[]
        for step in [0,1024]:
            group=diag[(diag.task==task)&(diag.step==step)].sort_values('seed')
            result=run.bootstrap(group.path_best_rank_one_capture)
            values.append(result);table.append(dict(panel='F',task=task,step=step,**result))
        means=np.array([z['mean'] for z in values]);lo=np.array([z['ci95_low'] for z in values]);hi=np.array([z['ci95_high'] for z in values])
        f.errorbar([0,1],means,yerr=[means-lo,hi-means],color=COLORS['ink'],marker=marker,ls=style,lw=LW_DATA,
            ms=4.6,capsize=2,label=TASK_NAMES[task],markerfacecolor='white' if task=='quartet' else COLORS['ink'])
    f.set_xlim(-.12,1.12);f.set_ylim(0,1.075);f.set_xticks([0,1],['Initial','Trained']);f.set_yticks([0,.5,1])
    f.set_ylabel('Best rank-one path capture')
    f.legend(loc='lower left',frameon=False,fontsize=PT_LEGEND,handlelength=2.,labelspacing=.2,borderaxespad=.15)
    path=dest/'credit_interaction_bridge_native.pdf'
    result=canvas.save(path,name='credit_interaction_bridge_native',dpi=200)
    pd.DataFrame(table).to_csv(dest/'figure_source.csv',index=False)
    inputs=[source/'all_curves.csv',source/'all_diagnostics.csv',source/'paired_contrasts.csv',source/'paired_seed_contrasts.csv',
            run.OUT/'protocol_freeze.json',run.OUT/'selection_freeze.json']
    run.write(dest/'figure_provenance.json',dict(figure=str(path.relative_to(run.JOURNAL)),
        builder_sha256=run.sha(Path(__file__)),sources_sha256={str(p.relative_to(run.JOURNAL)):run.sha(p) for p in inputs},
        figure_sha256=run.sha(path),n_seed_blocks=20,optimizer='Adam',panels='A–F',
        selection='B–D use frozen per-rule rates; E includes frozen common-rate control; F uses exact-rule selected rate',
        field='Unweighted six-site nonsomatic path; optimal current fixed rank-one direction; one oracle coefficient per example'))
    plt.close(canvas.fig)
    return result

if __name__=='__main__': build()
