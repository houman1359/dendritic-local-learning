#!/usr/bin/env python3
"""Native six-panel report of all MNIST ladder controls and coordinate audit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from journal_style import COLORS,PT_TICK,PT_LABEL,PT_TITLE,PT_LEGEND,PANEL_LABEL_PT,apply_neurips_style
from run import OUT
ARMS=('strict_scalar','neuron_shared','projected_k1','subtree_k3','exact_path','decoder_only')
LABELS={'strict_scalar':'Strict\nscalar','neuron_shared':'Neuron\nshared','projected_k1':'Projected\nK = 1','subtree_k3':'Subtree\nK = 3','exact_path':'Exact\npath','decoder_only':'Decoder\nonly'}
HUES=dict(zip(ARMS,[COLORS['scalar'],COLORS['per_soma'],COLORS['exc'],COLORS['pathway'],COLORS['bp'],COLORS['point_mlp']]))
apply_neurips_style()
plt.rcParams.update({'font.size':7.6,'axes.labelsize':8.4,'axes.titlesize':8.8,'xtick.labelsize':7.6,'ytick.labelsize':7.6,'legend.fontsize':7.4,'axes.linewidth':.7,'pdf.fonttype':42,'ps.fonttype':42})

def main():
    folder=OUT/'summaries';rates=pd.read_csv(folder/'development_rate_summary_six_rules.csv');fresh=pd.read_csv(folder/'fresh_analysis_rows_six_rules.csv')
    summary=pd.read_csv(folder/'condition_summary_six_rules.csv');contrast=pd.read_csv(folder/'paired_contrasts_six_rules.csv');capture=pd.read_csv(folder/'delivery_coordinate_capture_summary.csv')
    accuracy_min=max(0.,5*np.floor((100*fresh.test_accuracy.min()-3)/5))
    fig,axes=plt.subplots(3,2,figsize=(7.2,7.0))
    fig.subplots_adjust(left=.105,right=.978,bottom=.080,top=.937,wspace=.40,hspace=.64)
    for letter,ax in zip('ABCDEF',axes.flat):
        ax.spines[['top','right']].set_visible(False)
        ax.text(-.23,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=10.5,va='top')
    for col,architecture in enumerate(('shunting','additive')):
        ax=axes[0,col]
        for arm in ARMS:
            g=rates[rates.architecture.eq(architecture)&rates.arm.eq(arm)&rates.metric.eq('selection_validation_loss')].sort_values('multiplier')
            ax.errorbar(g.multiplier,g['mean'],yerr=[g['mean']-g.ci_low,g.ci_high-g['mean']],color=HUES[arm],marker='o',ms=3,lw=1.1,capsize=2,label=LABELS[arm].replace('\n',' '))
        ax.set(xscale='log',yscale='log',xticks=[.3,1,3],xticklabels=['0.3','1','3'],xlabel='Learning-rate multiplier',ylabel='Validation cross-entropy',title=f'{architecture.capitalize()}: development sweep')
        ax.grid(axis='y',alpha=.18)
        ax=axes[1,col]
        for policy,offset,filled in [('selected',-.10,True),('common_original',.10,False)]:
            for i,arm in enumerate(ARMS):
                g=fresh[fresh.architecture.eq(architecture)&fresh.arm.eq(arm)&fresh.rate_policy.eq(policy)]
                values=100*g.test_accuracy.to_numpy()
                ax.scatter(i+offset+np.linspace(-.04,.04,len(g)),values,color=HUES[arm],s=7,alpha=.28,zorder=2)
                r=summary[summary.architecture.eq(architecture)&summary.arm.eq(arm)&summary.rate_policy.eq(policy)&summary.metric.eq('test_accuracy')].iloc[0]
                ax.errorbar(i+offset,100*r['mean'],yerr=[[100*(r['mean']-r.ci_low)],[100*(r.ci_high-r['mean'])]],color=HUES[arm],marker='o' if filled else 's',mfc=HUES[arm] if filled else 'white',ms=4,capsize=2,lw=.9,zorder=3)
        ticklabels={**LABELS,'neuron_shared':'Per\nneuron','subtree_k3':'K = 3\nsubtrees'}
        ax.set(xticks=range(6),xticklabels=[ticklabels[x] for x in ARMS],ylabel='Test accuracy (%)',ylim=(accuracy_min,101),title=f'{architecture.capitalize()}: fresh paired seeds')
    handles=[plt.Line2D([],[],color=HUES[a],marker='o',ms=3,lw=1,label=LABELS[a].replace('\n',' ')) for a in ARMS]
    fig.legend(handles=handles,ncol=6,loc='upper center',bbox_to_anchor=(.53,1),frameon=False,columnspacing=1.1,handlelength=1.4)
    ax=axes[2,0];names=['subtree_k3_minus_projected_k1','exact_path_minus_subtree_k3']
    for architecture,archoffset in [('shunting',-.14),('additive',.14)]:
        for policy,policyoffset,marker,filled in [('selected',-.045,'o',True),('common_original',.045,'s',False)]:
            g=contrast[contrast.architecture.eq(architecture)&contrast.rate_policy.eq(policy)&contrast.metric.eq('test_accuracy')].set_index('contrast').loc[names]
            ax.errorbar(np.arange(2)+archoffset+policyoffset,100*g['mean'],yerr=[100*(g['mean']-g.ci_low),100*(g.ci_high-g['mean'])],color=COLORS[architecture],marker=marker,mfc=COLORS[architecture] if filled else 'white',ms=4,capsize=2,lw=.9,ls='none')
    ax.axhline(0,color='#777777',lw=.8,ls='--');ax.set(xticks=[0,1],xticklabels=['K = 3 − projected K = 1','Exact − K = 3'],ylabel='Paired accuracy difference (pp)',title='Added within-tree resolution')
    ax=axes[2,1]
    for architecture,offset in [('shunting',-.12),('additive',.12)]:
        for basis,marker,style in [('broadcast_k1','^','-'),('subtrees_k3','D','--')]:
            g=capture[capture.architecture.eq(architecture)&capture.coordinate.eq('activation')&capture.basis.eq(basis)&capture.metric.eq('mean_capture')].set_index('checkpoint').loc[['initial','trained']]
            ax.errorbar(np.arange(2)+offset,g['mean'],yerr=[g['mean']-g.ci_low,g.ci_high-g['mean']],color=COLORS[architecture],marker=marker,ls=style,ms=3.6,lw=1,capsize=2)
    ax.set(xticks=[0,1],xticklabels=['Initial','Trained'],ylim=(0,1.03),ylabel='Mean field-energy capture',title='Activation-space dictionary capture')
    foot=[plt.Line2D([],[],color='#444',marker='o',lw=0,label='Selected rate'),plt.Line2D([],[],color='#444',marker='s',mfc='white',lw=0,label='Original common rate')]
    foot += [plt.Line2D([],[],color=COLORS[a],lw=1.4,label=a.capitalize()) for a in ('shunting','additive')]
    foot2=[plt.Line2D([],[],color='#444',marker='^',lw=1,label='K = 1'),plt.Line2D([],[],color='#444',marker='D',lw=1,ls='--',label='K = 3')]
    fig.legend(handles=foot+foot2,ncol=6,loc='lower center',bbox_to_anchor=(.53,.004),frameon=False,fontsize=6.8,columnspacing=1.1,handlelength=1.1)
    target=OUT/'figures';target.mkdir(exist_ok=True)
    fig.savefig(target/'image_ladder_controls_native.pdf',metadata={'CreationDate':None,'ModDate':None,'Creator':'dendritic-credit-figure-builder'});fig.savefig(target/'image_ladder_controls_native.png',dpi=160);plt.close(fig)
    rows=[]
    for panel,source,data in [('A-B','development_rate_summary_six_rules.csv',rates),('C-D','condition_summary_six_rules.csv',summary),('C-D individual seeds','fresh_analysis_rows_six_rules.csv',fresh),('E','paired_contrasts_six_rules.csv',contrast),('F','delivery_coordinate_capture_summary.csv',capture)]:
        for row in data.to_dict('records'):rows.append(dict(panel=panel,source=source,**row))
    pd.DataFrame(rows).to_csv(target/'figure_source.csv',index=False)
    print(target/'image_ladder_controls_native.pdf')
if __name__=='__main__':main()
