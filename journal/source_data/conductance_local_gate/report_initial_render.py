#!/usr/bin/env python3
"""Complete prospective local-gate summaries and native vector figures."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import OUT,J,protocol,digest,write,utc

LABELS={'exact':'Exact path','unit_broadcast':'Unit broadcast','calibrated_broadcast':'Initial profile','ancestry_three_oracle':'Three-pattern oracle','hard_distal_unit_proximal':'Local distal gate','swapped_distal_unit_proximal':'Swapped gate','hard_distal_and_proximal':'Gate also proximal','ancestry_two_leaf_oracle_unit_proximal':'Two leaf patterns + unit proximal','shunt_proportional_unit_proximal':'Shunt-proportional gate'}
COLORS={'exact':'#222222','unit_broadcast':'#888888','calibrated_broadcast':'#CC6677','ancestry_three_oracle':'#0072B2','hard_distal_unit_proximal':'#009E73','swapped_distal_unit_proximal':'#D55E00','hard_distal_and_proximal':'#AA4499','ancestry_two_leaf_oracle_unit_proximal':'#56B4E9','shunt_proportional_unit_proximal':'#E69F00'}

def boot(values,rng,draws=20000):
    v=np.asarray(values,float);dist=v[rng.integers(len(v),size=(draws,len(v)))].mean(1)
    return np.quantile(dist,[.025,.975])

def signflip(v):
    v=np.asarray(v,float);n=len(v);assert n<=20;observed=abs(v.sum());hits=0
    for start in range(0,2**n,65536):
        bits=np.arange(start,min(start+65536,2**n),dtype=np.uint32)[:,None]
        signs=2*((bits>>np.arange(n,dtype=np.uint32))&1).astype(float)-1
        hits+=int(np.sum(np.abs(signs@v)>=observed-1e-12))
    return hits/2**n

def load_complete():
    cfg,freeze=protocol();replay=json.loads((OUT/'historical_replay/report.json').read_text());assert replay['status']=='PASS','Complete validated historical replay first'
    rows={k:[] for k in ['endpoints','curves','diagnostics']};allhash={}
    for seed in cfg['fresh_seeds']:
        p=OUT/'runs'/f'seed_{seed}_audit.json';a=json.loads(p.read_text());assert a['protocol_sha256']==freeze['protocol_sha256'];assert not a['canary_excluded']
        for f,h in a['files_sha256'].items():assert digest(p.parent/f)==h,f;allhash[str((p.parent/f).relative_to(OUT))]=h
        allhash[str(p.relative_to(OUT))]=digest(p)
        for k in rows:rows[k].append(pd.read_csv(p.parent/f'seed_{seed}_{k}.csv',float_precision='round_trip'))
    frames={k:pd.concat(v,ignore_index=True) for k,v in rows.items()}
    assert len(frames['endpoints'])==cfg['counts']['primary_and_extended_endpoint_views']
    assert not frames['endpoints'].duplicated(['seed','task','budget','rate','rule']).any()
    assert not frames['curves'].duplicated(['seed','task','step','rate','rule']).any()
    assert frames['endpoints'].groupby(['task','budget','rate','rule']).size().eq(20).all()
    return cfg,freeze,frames,allhash

def summarize(cfg,frames):
    rng=np.random.default_rng(cfg['statistics']['bootstrap_seed']);ep=frames['endpoints'];means=[];contrasts=[];seedrows=[]
    for (task,budget,rate,rule),f in ep.groupby(['task','budget','rate','rule'],sort=True):
        f=f.sort_values('seed');ci=boot(f.test_nmse,rng);means.append(dict(task=task,budget=budget,rate=rate,rule=rule,n=len(f),mean=f.test_nmse.mean(),median=f.test_nmse.median(),minimum=f.test_nmse.min(),maximum=f.test_nmse.max(),ci_low=ci[0],ci_high=ci[1],mean_fixed_endpoint=f.fixed_endpoint_test_nmse.mean(),max_best_step=f.best_step.max(),min_changed_parameters=f.n_changed_parameters.min(),inhibitory_gain0_changed=int(f.g_i0_changed.sum()),inhibitory_gain1_changed=int(f.g_i1_changed.sum())))
    pairs=[('unit_broadcast','hard_distal_unit_proximal'),('hard_distal_unit_proximal','exact'),('swapped_distal_unit_proximal','hard_distal_unit_proximal'),('hard_distal_and_proximal','hard_distal_unit_proximal'),('ancestry_two_leaf_oracle_unit_proximal','exact'),('shunt_proportional_unit_proximal','exact'),('ancestry_three_oracle','exact'),('calibrated_broadcast','unit_broadcast')]
    def add(values,task,budget,rate,left,right):
        values=values.reindex(cfg['fresh_seeds']);ci=boot(values,rng)
        rec=dict(task=task,budget=budget,rate=rate,left=left,right=right,n=len(values),mean=values.mean(),ci_low=ci[0],ci_high=ci[1],positive=int((values>0).sum()),negative=int((values<0).sum()))
        contrasts.append(rec)
        for seed,v in values.items():seedrows.append(dict(task=task,budget=budget,rate=rate,left=left,right=right,seed=seed,difference=v))
    for (task,budget,rate),f in ep.groupby(['task','budget','rate'],sort=True):
        pivot=f.pivot(index='seed',columns='rule',values='test_nmse')
        for a,b in pairs:add(pivot[a]-pivot[b],task,budget,rate,a,b)
    for (budget,rate),f in ep.groupby(['budget','rate'],sort=True):
        p=f.pivot(index='seed',columns=['task','rule'],values='test_nmse')
        for a,b in [('unit_broadcast','hard_distal_unit_proximal'),('calibrated_broadcast','exact')]:
            add((p[('opposed_strong',a)]-p[('opposed_strong',b)])-(p[('aligned_strong',a)]-p[('aligned_strong',b)]),'opposed_minus_aligned',budget,rate,a,b)
    ct=pd.DataFrame(contrasts);sr=pd.DataFrame(seedrows);primary=[]
    for task in ['opposed_strong','opposed_minus_aligned']:
        f=sr[(sr.task==task)&(sr.budget==4096)&(sr.rate==.03)&(sr.left=='unit_broadcast')&(sr.right=='hard_distal_unit_proximal')]
        primary.append(dict(task=task,contrast='unit_minus_local_gate',p=signflip(f.difference)))
    order=np.argsort([r['p'] for r in primary]);previous=0.
    for rank,idx in enumerate(order):previous=max(previous,min(1,(len(primary)-rank)*primary[idx]['p']));primary[idx]['holm_p']=previous
    hardgap=ct[(ct.task=='opposed_strong')&(ct.budget==4096)&(ct.rate==.03)&(ct.left=='hard_distal_unit_proximal')&(ct.right=='exact')].iloc[0]
    hard=ep[(ep.task=='opposed_strong')&(ep.budget==4096)&(ep.rate==.03)&(ep.rule=='hard_distal_unit_proximal')]
    practical=dict(hard_minus_exact_upper95=float(hardgap.ci_high),upper95_below_001=bool(hardgap.ci_high<.001),hard_maximum=float(hard.test_nmse.max()),every_seed_below_01=bool(hard.test_nmse.lt(.01).all()),scope='Prespecified descriptive practical-precision reference; not an asymptotic equivalence test.')
    return pd.DataFrame(means),ct,sr,primary,practical

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.2,'axes.labelsize':7.2,'axes.titlesize':8,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'legend.fontsize':6.8,'axes.linewidth':.7,'lines.linewidth':1.5,'xtick.major.width':.7,'ytick.major.width':.7,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})

def panel(ax,label):
    ax.text(-.14,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom');ax.spines[['top','right']].set_visible(False)

def save(fig,name):
    d=OUT/'figures';d.mkdir(exist_ok=True);fig.savefig(d/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None,'Creator':'conductance_local_gate/report.py'});fig.savefig(d/(name+'.png'),dpi=180);plt.close(fig)

def figures(cfg,frames,means,contrasts):
    style();rng=np.random.default_rng(119);ep=frames['endpoints'];cur=frames['curves']
    fig=plt.figure(figsize=(7.2,7.4));gs=fig.add_gridspec(3,2,left=.10,right=.98,bottom=.08,top=.90,hspace=.80,wspace=.43)
    axes=[fig.add_subplot(gs[i,j]) for i in range(3) for j in range(2)]
    selected=['exact','hard_distal_unit_proximal','unit_broadcast','ancestry_two_leaf_oracle_unit_proximal']
    ax=axes[0];panel(ax,'A');ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    pos={6:(.5,.84),4:(.22,.5),5:(.78,.5),0:(.08,.16),1:(.36,.16),2:(.64,.16),3:(.92,.16)}
    for c,p in [(0,4),(1,4),(2,5),(3,5),(4,6),(5,6)]:
        x,y=pos[c];xp,yp=pos[p];ax.plot([x,xp],[y,yp],color='#555555',lw=1.2,zorder=1)
    for n,(x,y) in pos.items():
        col='#EEEEEE' if n>=4 else ('#009E73' if n<2 else '#DDDDDD');ax.scatter(x,y,s=105 if n==6 else 75,c=col,edgecolors='#333333',linewidths=.8,zorder=2)
    ax.text(.5,.96,'Somatic error',ha='center',fontsize=6.8);ax.text(.21,.64,'Unit proximal credit',ha='center',fontsize=6.8);ax.text(.80,.64,'Unit proximal credit',ha='center',fontsize=6.8)
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
            ax.plot(gr.mean().index,gr.mean(),label=LABELS[rule],color=COLORS[rule])
        ax.set_yscale('log');ax.set_xscale('symlog',linthresh=64);ax.set_xlabel('Updates');ax.set_ylabel('Held-out NMSE');ax.set_xticks([0,256,4096,16384]);ax.set_xticklabels(['0','256','4,096','16,384']);ax.axvline(4096,color='#666666',ls=':',lw=.8)
    handles,labels=axes[3].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,loc='upper center',bbox_to_anchor=(.52,.997),fontsize=6.8,ncol=2)
    ax=axes[4];panel(ax,'E');ax.set_title('Task-dependent benefit of the local gate')
    for i,budget in enumerate([4096,16384]):
        p=ep[(ep.rate==.03)&(ep.budget==budget)].pivot(index='seed',columns=['task','rule'],values='test_nmse')
        vals=(p[('opposed_strong','unit_broadcast')]-p[('opposed_strong','hard_distal_unit_proximal')])-(p[('aligned_strong','unit_broadcast')]-p[('aligned_strong','hard_distal_unit_proximal')])
        m=contrasts[(contrasts.task=='opposed_minus_aligned')&(contrasts.rate==.03)&(contrasts.budget==budget)&(contrasts.left=='unit_broadcast')&(contrasts.right=='hard_distal_unit_proximal')].iloc[0]
        ax.scatter(np.full(len(vals),i)+rng.uniform(-.09,.09,len(vals)),vals,s=8,color='#009E73',alpha=.45)
        ax.errorbar(i,m['mean'],yerr=[[m['mean']-m.ci_low],[m.ci_high-m['mean']]],fmt='D',color='#007C5B',markersize=4,capsize=3)
    ax.axhline(0,color='#777777',ls=':',lw=.8);ax.set_xlim(-.5,1.5);ax.set_xticks([0,1],['4,096','16,384']);ax.set_xlabel('Validation window (updates)');ax.set_ylabel('Broadcast − local gate NMSE,\nopposed minus aligned')
    ax=axes[5];panel(ax,'F');ax.set_title('Which compartments receive the gate?')
    order=['exact','hard_distal_unit_proximal','swapped_distal_unit_proximal','hard_distal_and_proximal']
    for i,rule in enumerate(order):
        f=ep[(ep.task=='opposed_strong')&(ep.rate==.03)&(ep.budget==4096)&(ep.rule==rule)]
        m=means[(means.task=='opposed_strong')&(means.rate==.03)&(means.budget==4096)&(means.rule==rule)].iloc[0]
        ax.scatter(np.full(len(f),i)+rng.uniform(-.08,.08,len(f)),f.test_nmse,s=8,color=COLORS[rule],alpha=.5)
        ax.errorbar(i,m['mean'],yerr=[[m['mean']-m.ci_low],[m.ci_high-m['mean']]],fmt='D',color=COLORS[rule],markersize=4,capsize=2)
    ax.set_xticks(range(4),['Exact','Distal\ngate','Swapped\ngate','Gate also\nproximal']);ax.set_yscale('log');ax.set_ylabel('Opposed test NMSE');ax.set_ylim(bottom=min(ep.test_nmse.min()/2,1e-6))
    save(fig,'local_gate_primary')
    # Equal-rate robustness: one heat map per task and budget; all rules retained.
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.7),layout='constrained')
    for ax,(task,budget),letter in zip(axes.flat,[(t,b) for t in ['aligned_strong','opposed_strong'] for b in [4096,16384]],'ABCD'):
        panel(ax,letter);table=means[(means.task==task)&(means.budget==budget)].pivot(index='rule',columns='rate',values='mean').reindex(cfg['rules']);values=np.log10(table.to_numpy());im=ax.imshow(values,vmin=-8,vmax=0,aspect='auto',cmap='viridis');ax.set_yticks(range(len(table)),[LABELS[r] for r in table.index] if letter in 'AC' else ['']*len(table),fontsize=6.8);ax.set_xticks(range(len(table.columns)),[str(x) for x in table.columns]);ax.set_xlabel('Frozen Adam rate');ax.set_title(task.replace('_strong','').capitalize()+f'; {budget:,} updates')
        for i in range(len(table)):
            for j in range(len(table.columns)):ax.text(j,i,f'{table.iloc[i,j]:.1e}',ha='center',va='center',fontsize=6.8,color='white' if values[i,j]<-3 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='log10 mean test NMSE',shrink=.65);save(fig,'local_gate_all_rates')

def main():
    cfg,freeze,frames,inputs=load_complete();folder=OUT/'summaries';folder.mkdir(exist_ok=True)
    for k,f in frames.items():f.to_csv(folder/f'all_{k}.csv',index=False)
    means,contrasts,seeds,tests,practical=summarize(cfg,frames)
    for name,f in [('condition_means',means),('paired_contrasts',contrasts),('paired_seed_contrasts',seeds)]:f.to_csv(folder/f'{name}.csv',index=False)
    figures(cfg,frames,means,contrasts)
    primary=means[(means.rate==.03)&(means.budget==4096)].pivot(index='rule',columns='task',values='mean').reindex(cfg['rules'])
    lines=['# Prospective local conductance-gate follow-up','',f'Protocol SHA256: `{freeze["protocol_sha256"]}`. Twenty entirely fresh seed blocks; 1,080 trajectories; all continue to 16,384 updates. Prior panel exploratory results are excluded.','', '## Primary Adam 0.03 outcomes at the 4,096-update window','',primary.to_markdown(floatfmt='.7g'),'','## Primary paired tests','',pd.DataFrame(tests).to_markdown(index=False),'','## Prespecified practical precision reference','',json.dumps(practical,indent=2),'','These gates use externally supplied local inhibitory context and a somatic error. They do not discover an endogenous context signal or error. The two-leaf profile rule uses oracle projection coefficients and unit proximal credit; it is not a wholly two-column six-site dictionary. Gating the proximal credit deliberately freezes two inhibitory gains through its interaction with local eligibility. All such outcomes are retained.','', 'Complete selected/fixed endpoints, all rates, seed contrasts, parameters and source hashes accompany this report. These are finite-budget outcomes.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    outcomehash={str(p.relative_to(OUT)):digest(p) for p in sorted(folder.glob('*.csv'))}
    figsha={str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'figures').glob('*')) if p.is_file()}
    write(folder/'completeness_audit.json',dict(status='PASS',completed_utc=utc(),protocol_sha256=freeze['protocol_sha256'],independent_fresh_seeds=cfg['fresh_seeds'],n_trajectories=1080,n_endpoint_views=len(frames['endpoints']),n_curve_rows=len(frames['curves']),primary_tests=tests,practical_reference=practical,scientific_input_sha256=inputs,summary_sha256=outcomehash,figure_sha256=figsha,report_script_sha256=digest(Path(__file__)),matplotlib=matplotlib.__version__))
    print(primary.to_string());print(json.dumps(practical,indent=2));print('All20freshblocks complete and verified.')
if __name__=='__main__':main()
