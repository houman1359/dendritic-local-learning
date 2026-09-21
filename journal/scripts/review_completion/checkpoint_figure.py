"""Supplementary Figure S37: learned tuning and interaction surfaces."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from figure_canvas import NativeCanvas,Margins,COLORS,style_panel,LW_DATA,LW_HAIR
from checkpoint_computation import interaction_component
from journal_style import DIV_CMAP
J=Path(__file__).resolve().parents[2];D=J/'source_data/checkpoint_computation'

def main():
    c=NativeCanvas(446/72,3,row_weights=[110,106,106],hgutter_pt=22,vgutter_pt=34,
                   margins=Margins(left=38,right=50,top=29,bottom=34))
    tuning=pd.read_csv(D/'branch_tuning.csv');surfaces=pd.read_csv(D/'population_surfaces.csv')
    colours={'target':COLORS['mute'],'unit_broadcast':COLORS['scalar'],'hard_distal_unit_proximal':COLORS['shunting'],'exact':COLORS['bp']}
    names={'target':'Teacher','unit_broadcast':'Broadcast','hard_distal_unit_proximal':'Hard distal gate','exact':'Exact'}
    rng=np.random.default_rng(2026092121);indices=rng.integers(20,size=(10000,20));rows=[]
    for context,letter in enumerate('AB'):
        ax=c.panel(letter,0,context*6,colspan=6);style_panel(ax)
        for rule in ['unit_broadcast','hard_distal_unit_proximal','exact','target']:
            part=tuning[tuning.context.eq(context)&tuning.rule.eq(rule)]
            wide=part.pivot(index='seed',columns='z1',values='branch_contribution').sort_index();assert len(wide)==20
            x=wide.columns.to_numpy();values=wide.to_numpy();mean=values.mean(0)
            low,high=np.quantile(values[indices].mean(1),[.025,.975],axis=0)
            dash=(0,(3,2)) if rule=='target' else ((0,(1,2)) if rule=='exact' else '-')
            ax.plot(x,mean,color=colours[rule],ls=dash,lw=LW_DATA,zorder=5 if rule=='target' else 3)
            ax.fill_between(x,low,high,color=colours[rule],alpha=.12,lw=0)
            rows.extend(dict(panel=letter,rule=rule,z1=float(a),mean=float(b),ci95_low=float(lo),ci95_high=float(hi),quantity='selected branch contribution') for a,b,lo,hi in zip(x,mean,low,high))
        ax.set_xlim(-2,2);ax.set_xticks([-2,0,2]);ax.set_ylim(0,.55);ax.set_yticks([0,.25,.5]);ax.set_xlabel('Feature 1')
        ax.set_ylabel('Contribution to soma' if context==0 else '')
        ax.text(.5,1.08,'Selected left branch' if context==0 else 'Selected right branch',ha='center',transform=ax.transAxes,fontsize=8)
    c.fig.legend([Line2D([],[],color=colours[r],ls=(0,(3,2)) if r=='target' else ((0,(1,2)) if r=='exact' else '-'),lw=LW_DATA) for r in names],list(names.values()),loc='upper center',bbox_to_anchor=(.5,1.005),frameon=False,ncol=4,fontsize=7)
    grid=np.sort(surfaces.z1.unique());weights=np.ones(len(grid));weights[[0,-1]]=.5;weights/=weights.sum()
    z1,z2=np.meshgrid(grid,grid,indexing='ij');target=.5*(np.tanh(z1)+np.tanh(z2))+.25*np.tanh(z1)*np.tanh(z2)
    matrices={'target':target}
    for rule in ['resistance','derivative','exact']:
        part=surfaces[surfaces.rule.eq(rule)]
        assert part.seed.nunique()==20 and part.context.nunique()==4
        matrices[rule]=part.groupby(['z1','z2']).prediction.mean().unstack().loc[grid,grid].to_numpy()
    titles=['Target','Resistance gate h','Augmented h f′','Exact']
    images=[]
    for row,letters in [(1,'CDEF'),(2,'GHIJ')]:
        for column,((rule,full),letter,title) in enumerate(zip(matrices.items(),letters,titles)):
            matrix=full if row==1 else interaction_component(full,weights)
            ax=c.panel(letter,row,column*3,colspan=3);style_panel(ax)
            lim=1.2 if row==1 else .25
            im=ax.pcolormesh(grid,grid,matrix.T,cmap=DIV_CMAP,vmin=-lim,vmax=lim,shading='nearest',rasterized=False)
            ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2' if column==0 else '')
            ax.text(.5,1.10,title,ha='center',transform=ax.transAxes,fontsize=8)
            rows.extend(dict(panel=letter,rule=rule,z1=float(grid[i]),z2=float(grid[j]),value=float(matrix[i,j]),quantity='response' if row==1 else 'interaction') for i in range(len(grid)) for j in range(len(grid)))
        images.append(im)
    locks=c.lock_reserves()
    left=max(v[0] for v in locks.values())+2;right=max(v[1] for v in locks.values())+2
    for letter in 'ABCDEFGHIJ':c.declare_reserve(letter,left=left,right=right)
    c.lock_reserves()
    # Shared scales outside the data panels; fixed bounds retain all values.
    for row,im,label in [(1,images[0],'Response'),(2,images[1],'Interaction')]:
        box=c.axes['C' if row==1 else 'G'].get_position()
        cbax=c.fig.add_axes([.923,box.y0+.1*box.height,.012,.8*box.height])
        cb=c.fig.colorbar(im,cax=cbax,ticks=[-1.2,0,1.2] if row==1 else [-.25,0,.25]);cb.ax.tick_params(labelsize=7,width=LW_HAIR,length=2)
        cb.ax.set_ylabel(label,fontsize=8,labelpad=3)
    output=J/'figures/supplementary/curated/checkpoint_computation.pdf'
    findings=c.save(output,name='checkpoint_computation',dpi=180)
    pd.DataFrame(rows).to_csv(D/'figure_S37_plotted.csv',index=False)
    report=dict(figure='figS37',source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.csv')},
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cohorts='All twenty seeds from each original primary cohort; checkpoint-only',findings=[str(x) for x in findings])
    (D/'figure_S37_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
