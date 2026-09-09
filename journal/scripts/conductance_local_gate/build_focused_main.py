#!/usr/bin/env python3
"""Focused main Figure 5; retain immutable scientific and supplementary outputs."""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import report as historical
from report import style,panel,COLORS,LABELS,boot,OUT
from portable_contract import J,load_protocol,verify
sys.path.insert(0,str(J/'scripts/credit_first_figures'))
from focused_provenance import publish

def build(cfg,frames,means,contrasts,emit_main=False):
    style();rng=np.random.default_rng(119);ep=frames['endpoints'];cur=frames['curves']
    fig=plt.figure(figsize=(7.2,7.4));gs=fig.add_gridspec(3,2,left=.10,right=.98,bottom=.08,top=.90,hspace=.80,wspace=.43)
    axes=[fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
    band_rows=[];band_rng=np.random.default_rng(2026090830)
    selected=['exact','hard_distal_unit_proximal','unit_broadcast','ancestry_two_leaf_oracle_unit_proximal']
    ax=axes[0];panel(ax,'A');ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    pos={6:(.5,.84),4:(.22,.5),5:(.78,.5),0:(.08,.16),1:(.36,.16),2:(.64,.16),3:(.92,.16)}
    for c,p in [(0,4),(1,4),(2,5),(3,5),(4,6),(5,6)]:
        x,y=pos[c];xp,yp=pos[p];ax.plot([x,xp],[y,yp],color='#555555',lw=1.2,zorder=1)
    for n,(x,y) in pos.items():
        col='#EEEEEE' if n>=4 else ('#009E73' if n<2 else '#DDDDDD');ax.scatter(x,y,s=105 if n==6 else 75,c=col,edgecolors='#333333',linewidths=.8,zorder=2)
    ax.text(.5,.96,'Somatic error',ha='center',fontsize=6.8);ax.text(.21,.64,'Unit proximal credit',ha='center',fontsize=6.8,bbox=dict(facecolor='white',edgecolor='none',pad=.25));ax.text(.80,.64,'Unit proximal credit',ha='center',fontsize=6.8,bbox=dict(facecolor='white',edgecolor='none',pad=.25))
    ax.text(.03,.31,'Uninhibited',color='#007C5B',fontsize=6.8);ax.text(.71,.31,'Inhibited',color='#777777',fontsize=6.8)
    ax.text(.20,.015,'Distal gate = 1',ha='center',fontsize=6.8);ax.text(.79,.015,'Distal gate = 0',ha='center',fontsize=6.8)
    ax.set_title('Local gate; all 24 parameters available',pad=11)
    ax=axes[1];panel(ax,'B');ax.set_title('Nominal teacher tuning')
    z=np.linspace(-2,2,201);xp=np.exp(z);xm=np.exp(-z)
    increasing=(8*xp+.25*xm)/(1+8*xp+.25*xm+.25*xp+8*xm)
    decreasing=(.25*xp+8*xm)/(1+.25*xp+8*xm+8*xp+.25*xm)
    ax.plot(z,increasing,color='#0072B2',label='Aligned: both; opposed: first')
    ax.plot(z,decreasing,color='#D55E00',label='Opposed: second subtree')
    ax.set_xlabel('Latent feature');ax.set_ylabel('Terminal voltage');ax.set_yticks([0,.5,1]);ax.legend(frameon=False,fontsize=6.8,loc='center left',bbox_to_anchor=(-.04,-.46))
    for ax,task,letter in [(axes[2],'aligned_strong','C'),(axes[3],'opposed_strong','D')]:
        panel(ax,letter);ax.set_title('Aligned tuning' if task=='aligned_strong' else 'Opposed tuning')
        for rule in selected:
            f=cur[(cur.task==task)&(cur.rate==.03)&(cur.rule==rule)];gr=f.groupby('step').test_nmse
            points=[]
            for step,group in f.groupby('step',sort=True):
                ci=boot(group.sort_values('seed').test_nmse,band_rng,cfg['statistics']['bootstrap_draws']);points.append((step,group.test_nmse.mean(),ci[0],ci[1]));band_rows.append(dict(task=task,rule=rule,rate=.03,step=step,mean=group.test_nmse.mean(),ci_low=ci[0],ci_high=ci[1]))
            b=np.asarray(points);ax.fill_between(b[:,0],b[:,2],b[:,3],color=COLORS[rule],alpha=.14,lw=0)
            ax.plot(gr.mean().index,gr.mean(),label=LABELS[rule],color=COLORS[rule])
        ax.set_yscale('log');ax.set_xscale('symlog',linthresh=64);ax.set_xlim(0,16384);ax.set_xlabel('Updates');ax.set_ylabel('Held-out NMSE');ax.set_xticks([0,256,1024,16384]);ax.set_xticklabels(['0','256','1,024','16,384']);ax.get_xticklabels()[-1].set_ha('right');ax.axvline(4096,color='#666666',ls=':',lw=.8)
    handles,labels=axes[3].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,loc='upper center',bbox_to_anchor=(.52,.997),fontsize=6.8,ncol=2)
    ax=axes[4];panel(ax,'E');ax.set_title('Additional benefit on opposed targets')
    for i,budget in enumerate([4096,16384]):
        p=ep[(ep.rate==.03)&(ep.budget==budget)].pivot(index='seed',columns=['task','rule'],values='test_nmse')
        vals=(p[('opposed_strong','unit_broadcast')]-p[('opposed_strong','hard_distal_unit_proximal')])-(p[('aligned_strong','unit_broadcast')]-p[('aligned_strong','hard_distal_unit_proximal')])
        m=contrasts[(contrasts.task=='opposed_minus_aligned')&(contrasts.rate==.03)&(contrasts.budget==budget)&(contrasts.left=='unit_broadcast')&(contrasts.right=='hard_distal_unit_proximal')].iloc[0]
        ax.scatter(np.full(len(vals),i)+rng.uniform(-.09,.09,len(vals)),vals,s=8,color='#009E73',alpha=.45)
        ax.errorbar(i,m['mean'],yerr=[[m['mean']-m.ci_low],[m.ci_high-m['mean']]],fmt='D',color='#007C5B',markersize=4,capsize=3)
    ax.axhline(0,color='#777777',ls=':',lw=.8);ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],['4,096','16,384']);ax.set_xlabel('Validation window (updates)');ax.set_ylabel('Extra error reduction on\nopposed targets (NMSE)')
    ax=axes[5];panel(ax,'F');ax.set_title('Which compartments receive the gate?')
    order=['exact','hard_distal_unit_proximal','shunt_proportional_unit_proximal','swapped_distal_unit_proximal','hard_distal_and_proximal']
    positions=[0,.90,1.95,3.25,4.45]
    for i,rule in enumerate(order):
        f=ep[(ep.task=='opposed_strong')&(ep.rate==.03)&(ep.budget==4096)&(ep.rule==rule)]
        m=means[(means.task=='opposed_strong')&(means.rate==.03)&(means.budget==4096)&(means.rule==rule)].iloc[0]
        ax.scatter(np.full(len(f),positions[i])+rng.uniform(-.08,.08,len(f)),f.test_nmse,s=8,color=COLORS[rule],alpha=.5)
        ax.errorbar(positions[i],m['mean'],yerr=[[m['mean']-m.ci_low],[m.ci_high-m['mean']]],fmt='D',color=COLORS[rule],markersize=4,capsize=2)
    ax.set_xticks(positions,['Exact','Hard\ngate','Continuous\ngate','Swapped\ngate','Gate also\nproximal']);ax.set_yscale('log');ax.set_ylabel('Opposed test NMSE');ax.set_ylim(1e-6,2);ax.set_xlim(-.45,5.05);ax.tick_params(axis='x',labelsize=7)
    rows=[]
    for zz,up,down in zip(z,increasing,decreasing):
        rows.append(dict(panel='B',record='nominal teacher',latent_feature=zz,increasing=up,decreasing=down))
    for row in band_rows:
        rows.append(dict(panel='C' if row['task']=='aligned_strong' else 'D',record='curve_mean',**row))
    archived=pd.read_csv(OUT/'figures/curve_band_source.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(pd.DataFrame(band_rows),archived,check_exact=False,rtol=1e-12,atol=1e-14)
    for budget in [4096,16384]:
        p=ep[(ep.rate==.03)&(ep.budget==budget)].pivot(index='seed',columns=['task','rule'],values='test_nmse')
        vals=(p[('opposed_strong','unit_broadcast')]-p[('opposed_strong','hard_distal_unit_proximal')])-(p[('aligned_strong','unit_broadcast')]-p[('aligned_strong','hard_distal_unit_proximal')])
        mean=contrasts[(contrasts.task=='opposed_minus_aligned')&(contrasts.rate==.03)&(contrasts.budget==budget)&(contrasts.left=='unit_broadcast')&(contrasts.right=='hard_distal_unit_proximal')].iloc[0]
        np.testing.assert_allclose(vals.mean(),mean['mean'],atol=1e-14)
        rows.append(dict(panel='E',record='mean_contrast',**mean.to_dict()))
        rows.extend(dict(panel='E',record='paired_seed',budget=budget,seed=seed,difference=value) for seed,value in vals.items())
    for rule in order:
        selected=ep[(ep.task=='opposed_strong')&(ep.rate==.03)&(ep.budget==4096)&(ep.rule==rule)]
        assert len(selected)==20
        mean=means[(means.task=='opposed_strong')&(means.rate==.03)&(means.budget==4096)&(means.rule==rule)].iloc[0]
        np.testing.assert_allclose(selected.test_nmse.mean(),mean['mean'],atol=1e-14)
        rows.append(dict(panel='F',record='condition_mean',**mean.to_dict()))
        rows.extend(dict(panel='F',record='seed_endpoint',**row) for row in selected.to_dict('records'))
    output=J/'figures/components/focused_main_05.pdf'
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None,'Creator':'conductance_local_gate/build_focused_main.py'})
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)
    sources=[OUT/'summaries'/name for name in ['all_curves.csv','all_endpoints.csv','condition_means.csv','paired_contrasts.csv']]
    sources += [OUT/'figures/curve_band_source.csv',OUT/'protocol.json',OUT/'protocol_freeze.json',OUT/'summaries/completeness_audit.json']
    builders=[Path(__file__),Path(historical.__file__),HERE/'model.py',HERE/'portable_contract.py',J/'scripts/journal_style.py']
    panels={'A':'Unchanged supplied-context distal-gate schematic, with ungated proximal eligibility.',
            'B':'Unchanged nominal teacher tuning, independently perturbed in every actual teacher.',
            'C':'Aligned-task four-rule curves at Adam0.03; twenty original fresh seeds and unchanged bootstrap intervals.',
            'D':'Opposed-task curves under the identical protocol and readout definitions.',
            'E':'(Broadcast minus hard-gate NMSE on opposed tasks) minus (broadcast minus hard-gate NMSE on aligned tasks), at both validation windows; unchanged paired seed outcomes and intervals.',
            'F':'Existing primary-rate opposed exact/hard/continuous/swapped/proximal-gated outcomes; no new task or rate selection. All twenty original fresh seeds retained.'}
    publish(5,output,rows,sources,builders,panels,emit_main=emit_main,
            notes='Editorial promotion of the already tested continuous gate. Existing report.py, experimental summaries, original primary figure, full-rate SI sheet and historical completion gate remain unchanged.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-main',action='store_true')
    args=parser.parse_args()
    cfg,freeze,_=load_protocol()
    audit=__import__('json').loads((OUT/'summaries/completeness_audit.json').read_text())
    assert audit['status']=='PASS'
    for relative,original in audit['summary_sha256'].items():verify(OUT/relative,original)
    frames={key:pd.read_csv(OUT/'summaries'/f'all_{key}.csv',float_precision='round_trip') for key in ['curves','endpoints']}
    means=pd.read_csv(OUT/'summaries/condition_means.csv',float_precision='round_trip')
    contrasts=pd.read_csv(OUT/'summaries/paired_contrasts.csv',float_precision='round_trip')
    build(cfg,frames,means,contrasts,args.emit_main)

if __name__=='__main__':main()
