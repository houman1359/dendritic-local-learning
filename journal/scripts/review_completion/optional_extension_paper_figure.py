"""Supplementary display of sensitivity proxies and learned cue routing."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas,Margins,COLORS,style_panel,LW_DATA,LW_HAIR
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conductance_local_gate'))
from conductance_local_gate.figure import plain_log_ticks

J=Path(__file__).resolve().parents[2]

def main(args):
    ep=pd.read_csv(args.data/'endpoints.csv');summary=pd.read_csv(args.data/'summary.csv');routing=pd.read_csv(args.data/'routing.csv')
    rng=np.random.default_rng(2026092175);rows=[]
    c=NativeCanvas(465/72,2,hgutter_pt=23,vgutter_pt=48,margins=Margins(left=42,right=42,top=35,bottom=43))
    def extract(study,arm,policy,metric='test_nmse'):
        group=ep[ep.study.eq(study)&ep.arm.eq(arm)&ep.policy.eq(policy)].sort_values('seed')
        row=summary[summary.study.eq(study)&summary.arm.eq(arm)&summary.policy.eq(policy)&summary.metric.eq(metric)].iloc[0]
        assert len(group)==20
        return group,row
    def colour(arm):
        if arm=='exact':return COLORS['bp']
        if 'resistance' in arm:return COLORS['scalar']
        if any(x in arm for x in ['shuffle','mean','uniform','wrong']):return COLORS['mute']
        return COLORS['shunting']
    def draw(ax,letter,study,arm,policy,x,jitter):
        g,s=extract(study,arm,policy);xx=x+rng.uniform(-jitter,jitter,20);col=colour(arm)
        ax.scatter(xx,g.test_nmse,s=6,color=col,alpha=.65,linewidths=0,zorder=4)
        ax.errorbar(x,s['mean'],yerr=[[s['mean']-s.ci95_low],[s.ci95_high-s['mean']]],fmt='D',mfc='white',mec=col,ecolor=col,ms=4,lw=.8,zorder=3)
        rows.extend(dict(panel=letter,study=study,arm=arm,policy=policy,seed=r.seed,x=xx[k],value=r.test_nmse,quantity='ordinary-test NMSE') for k,r in enumerate(g.itertuples()))
        return s['mean']
    ax=c.panel('A',0,0,colspan=6);style_panel(ax);ax.set_yscale('log')
    xx=[0,.25,.5,1.];means=[draw(ax,'A','proxy',a,'common',x,.012) for a,x in zip(['derivative','noise025','noise05','noise1'],xx)]
    ax.plot(xx,means,color=COLORS['shunting'],lw=LW_DATA,zorder=1)
    for arm,ls,label in [('resistance','--','Resistance gate'),('shuffle_noise05',':','Shuffled noise 0.5')]:
        g,s=extract('proxy',arm,'common');ax.axhline(s['mean'],color=colour(arm),ls=ls,lw=LW_DATA,label=label)
        rows.append(dict(panel='A',study='proxy',arm=arm,policy='common',value=s['mean'],quantity='mean ordinary-test NMSE'))
    c.fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.002),frameon=False,ncol=2,fontsize=7)
    ax.set_xticks(xx);ax.set_xlim(-.06,1.06);ax.set_xlabel('Parent-voltage noise SD');ax.set_ylabel('Ordinary-test NMSE')
    ax=c.panel('B',0,6,colspan=6);style_panel(ax);ax.set_yscale('log')
    arms=['derivative','bins2','bins4','shuffle_bins4','mean_bins4','exact']
    labels=['Exact\nslope','2 bins','4 bins','Shuffled\n4 bins','Mean\n4 bins','Exact\ncredit']
    for k,a in enumerate(arms):draw(ax,'B','proxy',a,'common',k,.12)
    ax.set_xticks(range(len(arms)),labels,fontsize=7);ax.set_xlim(-.55,len(arms)-.45)
    ax.set_ylabel('Ordinary-test NMSE')
    ax=c.panel('C',1,0,colspan=7);style_panel(ax);ax.set_yscale('log')
    arms=['oracle_augmented','learned_local_augmented','learned_exact_router_augmented','learned_local_resistance','uniform_augmented','wrong_augmented']
    labels=['Supplied\nroute','Learned\nlocal','Exact\ncontroller','Learned\nresistance','Uniform\nroute','Wrong\nroute']
    for k,a in enumerate(arms):draw(ax,'C','routing',a,'selected',k,.12)
    ax.set_xticks(range(len(arms)),labels,fontsize=7);ax.set_xlim(-.55,len(arms)-.45);ax.set_ylabel('Ordinary-test NMSE')
    ax=c.panel('D',1,7,colspan=5);style_panel(ax)
    matrices=[]
    for arm in ['learned_local_augmented','learned_local_resistance']:
        g=routing[routing.arm.eq(arm)&routing.policy.eq('selected')]
        matrix=g.groupby(['context','branch']).probability.mean().unstack().to_numpy();matrices.append(matrix)
        rows.extend(dict(panel='D',study='routing',arm=arm,policy='selected',context=i,branch=j,value=matrix[i,j],quantity='mean route probability') for i in range(4) for j in range(4))
    display=np.column_stack([matrices[0],np.full(4,np.nan),matrices[1]])
    im=ax.pcolormesh(np.arange(10)-.5,np.arange(5)-.5,display,vmin=0,vmax=1,cmap='Greys',shading='flat')
    ax.set_ylim(3.5,-.5);ax.set_xlim(-.5,8.5)
    ax.set_xticks([0,1,2,3,5,6,7,8],[0,1,2,3,0,1,2,3]);ax.set_yticks(range(4));ax.set_xlabel('Disinhibited stream');ax.set_ylabel('Cue')
    ax.text(2/9,1.04,'Augmented',ha='center',transform=ax.transAxes,fontsize=7)
    ax.text(7/9,1.04,'Resistance',ha='center',transform=ax.transAxes,fontsize=7)
    low=min(c.axes[l].get_ylim()[0] for l in 'AB');high=max(c.axes[l].get_ylim()[1] for l in 'AB')
    for letter in 'AB':c.axes[letter].set_ylim(low,high)
    for letter in 'AB':
        plain_log_ticks(c.axes[letter],[1e-5,1e-4,1e-3,1e-2,1e-1]);c.axes[letter].set_ylim(1e-5,.1)
    plain_log_ticks(c.axes['C'],[1e-5,1e-4,1e-3,1e-2,1e-1,1.]);c.axes['C'].set_ylim(5e-6,1.5)
    locks=c.lock_reserves();left=max(v[0] for v in locks.values())+2;right=max(v[1] for v in locks.values())+2
    for letter in 'ABCD':c.declare_reserve(letter,left=left,right=right)
    c.lock_reserves();box=c.axes['D'].get_position()
    cbax=c.fig.add_axes([.937,box.y0+.08*box.height,.009,.84*box.height]);cb=c.fig.colorbar(im,cax=cbax,ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=7,width=LW_HAIR,length=2);cb.ax.set_ylabel('Route probability',fontsize=8,labelpad=2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    findings=c.save(args.output,name='optional_extensions',dpi=180)
    pd.DataFrame(rows).to_csv(args.data/'figure_S38_plotted.csv',index=False)
    (args.data/'figure_S38_provenance.json').write_text(json.dumps(dict(figure='figS38',generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={name:hashlib.sha256((args.data/name).read_bytes()).hexdigest() for name in ['endpoints.csv','summary.csv','routing.csv']},
        findings=[str(x) for x in findings],scope='Two separate twenty-seed prospective cohorts; recurrent study deferred before training'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=J/'source_data/optional_extensions')
    p.add_argument('--output',type=Path,default=J/'figures/supplementary/curated/optional_extensions.pdf');main(p.parse_args())
