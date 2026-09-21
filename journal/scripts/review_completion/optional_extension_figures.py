"""Standalone extension figures from the complete, frozen fresh-seed cohorts."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas,Margins,COLORS,style_panel,LW_DATA

NAMES={'resistance':'Resistance','derivative':'Exact slope','exact':'Exact credit',
       'noise025':'Noise 0.25','noise05':'Noise 0.5','noise1':'Noise 1',
       'bins2':'2 bins','bins4':'4 bins','shuffle_bins4':'Shuffled\n4 bins','mean_bins4':'Mean\n4 bins',
       'oracle_augmented':'Supplied\nroute','learned_local_augmented':'Learned\nlocal route',
       'learned_exact_router_augmented':'Learned route\nexact error','learned_local_resistance':'Learned route\nresistance',
       'uniform_augmented':'Uniform\nroute','wrong_augmented':'Wrong\nroute',
       'exact_trace':'Exact\ntrace','augmented_trace':'Augmented\ntrace','resistance_trace':'Resistance\ntrace',
       'exact_one_step':'Exact\none-step','augmented_one_step':'Augmented\none-step','exact_no_memory':'No\nmemory'}

def colour(arm):
    if arm=='exact' or arm.startswith('exact_'):return COLORS['bp']
    if 'resistance' in arm:return COLORS['scalar']
    if 'shuffle' in arm or 'wrong' in arm:return COLORS['mute']
    return COLORS['shunting']

def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    ep=pd.read_csv(args.data/'endpoints.csv');summary=pd.read_csv(args.data/'summary.csv')
    outputs=[];plotted=[];rng=np.random.default_rng(2026092175)
    def points(ax,study,arms,policy,panel):
        style_panel(ax);ax.set_yscale('log')
        for x,arm in enumerate(arms):
            g=ep[ep.study.eq(study)&ep.arm.eq(arm)&ep.policy.eq(policy)].sort_values('seed')
            s=summary[summary.study.eq(study)&summary.arm.eq(arm)&summary.policy.eq(policy)&summary.metric.eq('test_nmse')].iloc[0]
            assert len(g)==20
            jitter=rng.uniform(-.12,.12,20);col=colour(arm)
            ax.scatter(x+jitter,g.test_nmse,s=6,c=col,alpha=.65,zorder=4,linewidths=0)
            ax.errorbar(x,s['mean'],yerr=[[s['mean']-s.ci95_low],[s.ci95_high-s['mean']]],fmt='D',ms=4,mfc='white',mec=col,ecolor=col,lw=.8,zorder=3)
            plotted.extend(dict(figure=study,panel=panel,arm=arm,policy=policy,seed=r.seed,x=x+jitter[k],value=r.test_nmse,quantity='ordinary-test NMSE') for k,r in enumerate(g.itertuples()))
        ax.set_xticks(range(len(arms)),[NAMES[a] for a in arms],fontsize=7)
        ax.set_xlim(-.55,len(arms)-.45);ax.set_ylabel('Ordinary-test NMSE');ax.grid(axis='y',alpha=.12)
    def save(canvas,name):
        locks=canvas.lock_reserves();left=max(v[0] for v in locks.values())+2;right=max(v[1] for v in locks.values())+2
        for letter in canvas.axes:canvas.declare_reserve(letter,left=left,right=right)
        canvas.lock_reserves();findings=canvas.save(args.output/(name+'.pdf'),name=name,dpi=180)
        outputs.append(dict(name=name,findings=[str(x) for x in findings]))

    c=NativeCanvas(245/72,1,hgutter_pt=24,margins=Margins(left=40,right=12,top=36,bottom=39))
    ax=c.panel('A',0,0,colspan=5);style_panel(ax);ax.set_yscale('log')
    arms=['derivative','noise025','noise05','noise1'];xx=[0,.25,.5,1.]
    means=[]
    for x,arm in zip(xx,arms):
        g=ep[ep.study.eq('proxy')&ep.arm.eq(arm)&ep.policy.eq('common')].sort_values('seed')
        s=summary[summary.study.eq('proxy')&summary.arm.eq(arm)&summary.policy.eq('common')&summary.metric.eq('test_nmse')].iloc[0]
        means.append(s['mean']);ax.scatter(x+rng.uniform(-.012,.012,20),g.test_nmse,s=6,c=COLORS['shunting'],alpha=.65,zorder=4,linewidths=0)
        ax.errorbar(x,s['mean'],yerr=[[s['mean']-s.ci95_low],[s.ci95_high-s['mean']]],fmt='D',ms=4,mfc='white',mec=COLORS['shunting'],ecolor=COLORS['shunting'],zorder=3)
        plotted.extend(dict(figure='proxy',panel='A',arm=arm,policy='common',seed=r.seed,x=x,value=r.test_nmse,quantity='ordinary-test NMSE') for r in g.itertuples())
    ax.plot(xx,means,color=COLORS['shunting'],lw=LW_DATA,zorder=1)
    for arm,style,label in [('resistance','--','Resistance'),('shuffle_noise05',':','Shuffled noise 0.5')]:
        value=summary.query("study=='proxy' and policy=='common' and metric=='test_nmse' and arm==@arm")['mean'].iloc[0]
        ax.axhline(value,color=colour(arm),ls=style,lw=LW_DATA,label=label)
        plotted.append(dict(figure='proxy',panel='A',arm=arm,policy='common',value=value,quantity='mean ordinary-test NMSE'))
    c.fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.005),frameon=False,fontsize=7,ncol=2)
    ax.set_xticks(xx);ax.set_xlim(-.06,1.07)
    ax.set_xlabel('Parent-voltage noise SD');ax.set_ylabel('Ordinary-test NMSE')
    ax=c.panel('B',0,5,colspan=7)
    points(ax,'proxy',['derivative','bins2','bins4','shuffle_bins4','mean_bins4','exact'],'common','B');ax.set_ylabel('')
    save(c,'proxy_robustness')

    c=NativeCanvas(240/72,1,hgutter_pt=24,margins=Margins(left=40,right=24,top=22,bottom=44))
    ax=c.panel('A',0,0,colspan=12)
    arms=['oracle_augmented','learned_local_augmented','learned_exact_router_augmented','learned_local_resistance','uniform_augmented','wrong_augmented']
    points(ax,'routing',arms,'selected','A')
    routes=pd.read_csv(args.data/'routing.csv')
    save(c,'learned_routing_error')
    c=NativeCanvas(245/72,1,hgutter_pt=40,margins=Margins(left=40,right=34,top=30,bottom=36))
    for letter,column,arm in [('A',0,'learned_local_augmented'),('B',6,'learned_exact_router_augmented')]:
        ax=c.panel(letter,0,column,colspan=6);style_panel(ax)
        g=routes[routes.arm.eq(arm)&routes.policy.eq('selected')]
        matrix=g.groupby(['context','branch']).probability.mean().unstack().to_numpy()
        im=ax.pcolormesh(np.arange(5)-.5,np.arange(5)-.5,matrix,vmin=0,vmax=1,cmap='Greys',shading='flat')
        ax.set_ylim(3.5,-.5);ax.set_xlim(-.5,3.5)
        ax.set_xticks(range(4));ax.set_yticks(range(4));ax.set_xlabel('Disinhibited input stream');ax.set_ylabel('Cue' if letter=='A' else '')
        ax.text(.5,1.08,'Local router update' if letter=='A' else 'Exact router update',transform=ax.transAxes,ha='center',fontsize=8)
        for cue in range(4):
            for branch in range(4):
                value=matrix[cue,branch];ax.text(branch,cue,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if value>.55 else 'black')
                plotted.append(dict(figure='routing',panel='map_'+letter,arm=arm,policy='selected',context=cue,branch=branch,value=value,quantity='mean route probability'))
    save(c,'learned_routing_maps')

    # Recurrent training was deferred before any such fit started.
    assert not ep.study.eq('temporal').any()
    pd.DataFrame(plotted).to_csv(args.output/'plotted.csv',index=False)
    (args.output/'provenance.json').write_text(json.dumps(dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.glob('*.csv')},figures=outputs),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
