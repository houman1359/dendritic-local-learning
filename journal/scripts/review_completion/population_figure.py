"""Main Figure 6: population task, assignment, failure and sensitivity rescue."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas, Margins, COLORS, style_panel

J=Path(__file__).resolve().parents[2]
D=J/'source_data/curated_publication'
OUTPUT=J/'figures/main/figure_06.pdf'
INPUTS=['inhibitory_selection_endpoints.csv','inhibitory_selection_summary.csv',
        'inhibitory_selection_rate_sensitivity.csv','inhibitory_rescue_endpoints.csv',
        'inhibitory_rescue_common_endpoints.csv','nonlinear_separable_endpoints.csv']
SOURCES={f'main_6{p}':['source_data/curated_publication/'+n for n in INPUTS]
         for p in 'BCDEFG'}
SCOPE={f'main_6{p}':s for p,s in zip('BCDEFG',[
    'Separable target, shunting forward: all five spatial-assignment rules under distractor stress; original paired seed cohort.',
    'Separable target at severity three: shunting, tonic and current forward controls, three credit rules.',
    'Broadcast and resistance at ordinary test and severity three; rate-selected and common-rate 0.1 sensitivity, same seed blocks.',
    'Nonlinear-parent interaction target: original-bound Adam rescue at separately selected rates; twenty new paired seed blocks.',
    'Same rescue seed blocks at common Adam rate 0.03; a sensitivity, not independent replication.',
    'Nonlinear parents with separable target at common Adam rate .03; same rescue seed blocks, only the target interaction removed.'])}
C={'exact':COLORS['bp'],'broadcast':COLORS['scalar'],'resistance':COLORS['shunting'],
   'swapped':COLORS['point_mlp'],'uniform_rms':COLORS['mute'],
   'derivative':'#147D92','shuffled_derivative':'#9E6584'}
LABEL={'exact':'Exact','broadcast':'Broadcast','resistance':'h','derivative':"h f′",
       'shuffled_derivative':'Shuffled f′','swapped':'Wrong branch','uniform_rms':'Uniform RMS'}
ROWS=[]


def boot(v):
    v=np.asarray(v);rng=np.random.default_rng(2026092103)
    bs=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return v.mean(),*np.quantile(bs,[.025,.975])


def point(ax,x,values,rule,panel,**meta):
    values=values.sort_values('seed'); v=values['value'].to_numpy()
    m,lo,hi=boot(v)
    ax.scatter(x+np.linspace(-.06,.06,len(v)),v,s=5,color=C[rule],alpha=.3,lw=0,zorder=2)
    ax.errorbar(x,m,yerr=[[m-lo],[hi-m]],fmt='D',ms=4.2,mfc='white',mec=C[rule],
                color=C[rule],elinewidth=.85,capsize=2,zorder=4)
    ROWS.append(dict(panel=panel,record='mean and bootstrap interval',rule=rule,mean=m,ci_low=lo,ci_high=hi,n=len(v),**meta))
    ROWS.extend(dict(panel=panel,record='seed outcome',rule=rule,seed=int(r.seed),value=r.value,**meta) for r in values.itertuples())


def style(ax,title,log_ticks):
    style_panel(ax);ax.set_ylabel('NMSE',fontsize=7)
    ax.set_title(title,fontsize=8,pad=9,loc='left')
    log_ticks(ax,[1e-5,1e-3,1e-1,1.]);ax.set_ylim(1e-5,1.5)


def task_panel(ax):
    """One illustrative tree; input streams map to its four parent branches."""
    from matplotlib.patches import Circle
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_axis_off()
    ax.text(.02,1.02,'Four streams × two features',fontsize=8,ha='left',va='bottom')
    xs=[.055,.135,.215,.295]
    for b,x in enumerate(xs):
        color=C['resistance'] if b==1 else C['uniform_rms']
        for off in [-.016,.016]:
            ax.plot([x+off,x],[.80,.47],color=color,lw=.85)
            ax.scatter([x+off],[.80],s=12,color=COLORS['exc'],edgecolor='none',zorder=3)
        ax.plot([x,.18],[.47,.10],color=color,lw=.85)
        ax.scatter([x],[.47],s=19,facecolor='white',edgecolor=color,lw=.85,zorder=4)
        if b!=1:
            ax.scatter([x+.018],[.51],s=10,color=COLORS['inh'],zorder=5)
    ax.scatter([.18],[.10],s=26,color=COLORS['ink'],zorder=4)
    ax.text(.345,.78,'Cue selects a stream',fontsize=7,ha='left')
    ax.text(.345,.48,'Others receive inhibition',fontsize=7,ha='left')
    ax.text(.345,.18,'Identity or tanh parents',fontsize=7,ha='left')
    ax.annotate('',xy=(.68,.36),xytext=(.60,.36),arrowprops=dict(arrowstyle='->',lw=.85,color=COLORS['ink']))
    ax.text(.715,.65,'16 neurons → readout',fontsize=8,ha='left')
    ax.text(.715,.27,'Target: selected stream',fontsize=7,ha='left')
    ax.text(.715,-.03,'Stress changes distractors only',fontsize=7,ha='left')


def build(log_ticks):
    ROWS.clear()
    ep=pd.read_csv(D/INPUTS[0]); summary=pd.read_csv(D/INPUTS[1])
    rate=pd.read_csv(D/INPUTS[2]); rescue=pd.read_csv(D/INPUTS[3]);common=pd.read_csv(D/INPUTS[4])
    original=ep[ep.variant.eq('separable')]
    c=NativeCanvas(552/72,4,row_weights=[57,91,91,91],hgutter_pt=30,vgutter_pt=53,
                   margins=Margins(left=46,right=12,top=26,bottom=37))
    c.letter_dx=27
    task=c.panel('A',0,0,12,letter='A',schematic=True,lock=False)
    task_panel(task)
    letters=dict(zip('HIJKLM','BCDEFG'))
    axes={p:c.panel(letters[p],1+i//2,(i%2)*6,6,letter=letters[p],grid='none') for i,p in enumerate('HIJKLM')}
    for p in axes:c.declare_reserve(letters[p],left=28,right=3)
    c.lock_reserves()
    titles=['Assignment under distractor stress','Forward inhibition, severity 3',
            'Rate and stress dependence','Nonlinear task: sensitivity rescue',
            'Interaction task: common rate 0.03','Same parents, separable target']
    for p,title in zip('HIJKLM',titles):style(axes[p],title,log_ticks)
    h=axes['H'];sev=[1.,1.5,2.,2.5,3.]
    for rule in ['exact','broadcast','resistance','swapped','uniform_rms']:
        part=summary[summary.variant.eq('separable')&summary.forward.eq('shunt')&summary.rule.eq(rule)].set_index('metric').loc[[f'ood_{s}' for s in sev]]
        h.plot(sev,part['mean'],color=C[rule],lw=1.05,label=LABEL[rule],ls='--' if rule=='resistance' else '-',marker='o',ms=2.2)
        h.fill_between(sev,part.ci_low,part.ci_high,color=C[rule],alpha=.12,lw=0)
        ROWS.extend(dict(panel='H',record='stress mean',rule=rule,severity=s,mean=r['mean'],ci_low=r.ci_low,ci_high=r.ci_high,n=20) for s,(_,r) in zip(sev,part.iterrows()))
    h.set_xlim(.9,3.1);h.set_xticks([1,2,3]);h.set_xlabel('Irrelevant-input severity',fontsize=7)
    h.legend(frameon=False,fontsize=7,ncol=2,loc='upper left',columnspacing=.6,handlelength=1.3,borderaxespad=.2)
    for k,forward in enumerate(['shunt','tonic','current']):
        for offset,rule in zip([-.23,0,.23],['exact','broadcast','resistance']):
            g=original[original.forward.eq(forward)&original.rule.eq(rule)]
            point(axes['I'],k+offset,g.assign(value=g['ood_3.0']),rule,'I',forward=forward,metric='ood_3.0')
    axes['I'].set_xticks(range(3),['Shunt','Tonic','Current']);axes['I'].set_xlim(-.5,2.5)
    axes['I'].legend([__import__('matplotlib').lines.Line2D([],[],color=C[r],marker='D',mfc='white',lw=0,ms=4) for r in ['exact','broadcast','resistance']],['Exact','Broadcast','h'],frameon=False,fontsize=7,ncol=3,loc='lower right',handlelength=.6,columnspacing=.7)
    for k,metric in enumerate(['test_nmse','ood_3.0','ood_3.0']):
        for offset,rule in zip([-.15,.15],['broadcast','resistance']):
            g=original[original.forward.eq('shunt')&original.rule.eq(rule)]
            if k==2 and rule=='broadcast':g=rate[rate.rule.eq(rule)&rate.rate.eq(.1)].rename(columns={'ood_3':'ood_3.0'})
            point(axes['J'],k+offset,g.assign(value=g[metric]),rule,'J',comparison=k,metric=metric,rate_scope='common 0.1' if k==2 else 'selected')
    axes['J'].set_xticks(range(3),['Ordinary','Stress 3','Stress 3\nrate 0.1']);axes['J'].set_xlim(-.5,2.5)
    for k,label in enumerate(['1.7×','87×','6.6×']):axes['J'].text(k,.23,label,ha='center',fontsize=7,color=COLORS['ink'])
    order=['exact','broadcast','resistance','derivative','shuffled_derivative']
    for panel,frame in [('K',rescue[rescue.optimizer.eq('adam')&rescue.bound.eq(9)]),('L',common)]:
        for k,rule in enumerate(order):
            g=frame[frame.rule.eq(rule)];point(axes[panel],k,g.assign(value=g.test_nmse),rule,panel,metric='test_nmse')
        axes[panel].set_xticks(range(5),['Exact','Broad-\ncast','h',"h f′",'Shuffled\nf′']);axes[panel].set_xlim(-.5,4.5)
    separable=pd.read_csv(D/'nonlinear_separable_endpoints.csv')
    for k,rule in enumerate(order):
        g=separable[separable.rule.eq(rule)]
        point(axes['M'],k,g.assign(value=g.test_nmse),rule,'M',metric='test_nmse',task='nonlinear parents, separable target')
    axes['M'].set_xticks(range(5),['Exact','Broad-\ncast','h',"h f′",'Shuffled\nf′']);axes['M'].set_xlim(-.5,4.5)
    for ax in axes.values():ax.tick_params(axis='x',labelsize=7,pad=3)
    findings=list(c.save(OUTPUT,name='figure_06',dpi=180))
    for row in ROWS: row['panel']=letters[row['panel']]
    from credit_first_figures.focused_provenance import publish
    SOURCES['main_6A']=[]
    SCOPE['main_6A']='Schematic of the supplied cue, four two-feature streams,16 [4,2] DendriNet neurons and linear readout; no measured series.'
    publish(6,OUTPUT,ROWS,[D/n for n in INPUTS],
            [Path(__file__),J/'scripts/conductance_local_gate/figure.py',J/'scripts/figure_canvas.py'],
            {k:dict(sources=SOURCES[k],scope=SCOPE[k]) for k in SOURCES},
            emit_main=False,layout_findings=findings,
            notes='Original and rescue cohorts are distinct; common-rate and separable-task sensitivities reuse rescue seeds. Rendering only.')
    return findings


def display_rows():
    return pd.read_csv(D/'figure_06_plotted.csv').to_dict('records')


if __name__ == '__main__':
    sys.path.insert(0, str(J/'scripts/conductance_local_gate'))
    from conductance_local_gate.figure import plain_log_ticks
    build(plain_log_ticks)
