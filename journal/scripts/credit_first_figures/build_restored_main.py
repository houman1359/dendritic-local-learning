#!/usr/bin/env python3
"""Render the restored task-to-credit main figures from immutable evidence.

Only figures/components, figures/provenance and (with --emit-main) figures/main
are written. No experiment, fit, endpoint selection or Source Data file is
changed. Whole-seed intervals are descriptive redraws of existing trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J/'scripts'))
import build_main_figure_06 as depth
import build_main_figure_07 as anatomy
import build_framework as framework
import build_anatomy as commonmode
import build_measured as measured
from figure_canvas import (NativeCanvas, Margins, COLORS, PT_LABEL, PT_ANNOT,
                           PT_SMALL, PT_LEGEND, LW_DATA, LW_REF, LW_EDGE,
                           LW_ERR, LW_HAIR, MARKER_MS, style_panel)
from journal_style import style_direct_color_labels

S=J/'source_data'
OUT=J/'figures/components'
REC=J/'figures/provenance/structure_restoration_20260908'
RULES=('exact','unit_broadcast','calibrated_broadcast')
RC={'exact':COLORS['bp'],'unit_broadcast':COLORS['scalar'],
    'calibrated_broadcast':COLORS['additive']}
RN={'exact':'Exact path','unit_broadcast':'Unit broadcast',
    'calibrated_broadcast':'Initial profile'}
TN={'matching':'Pairwise','quartet':'Quartic','nested':'Nested'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(S/path, float_precision='round_trip')


def boot(values, seed=210999):
    values=np.asarray(values,float)
    ix=np.random.default_rng(seed).integers(len(values),size=(10000,len(values)))
    means=values[ix].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975],axis=0)
    return values.mean(axis=0),lo,hi


def save(canvas, number, sources, panels, caption, rows=(), extra=None):
    OUT.mkdir(exist_ok=True);REC.mkdir(parents=True,exist_ok=True)
    path=OUT/f'restored_main_{number:02d}.pdf'
    style_direct_color_labels(canvas.fig)
    # Equal module spans keep equal plotting widths despite long forest labels.
    locks=canvas.lock_reserves()
    groups={}
    for rec in canvas._records:
        if not rec.get('schematic'):
            groups.setdefault(rec['colspan'],[]).append(rec['name'])
    for group in groups.values():
        if len(group)<2:continue
        left=max(locks[name][0]for name in group)
        right=max(locks[name][1]for name in group)
        for name in group:canvas.declare_reserve(name,left=left,right=right)
    findings=canvas.save(path,name=f'restored_main_{number:02d}',dpi=180)
    plt.close(canvas.fig)
    pd.DataFrame(rows).to_csv(REC/f'figure_{number:02d}_plotted.csv',index=False)
    helpers=[Path(__file__),J/'scripts/figure_canvas.py',J/'scripts/journal_style.py']
    provenance=dict(figure=f'Figure {number}',output=str(path.relative_to(J)),
        output_sha256=sha(path),panel_sources=panels,
        source_sha256={str((S/p).relative_to(J)):sha(S/p) for p in sources},
        builder_sha256={str(p.relative_to(J)):sha(p) for p in helpers},
        source_data_immutable=True,new_experiments=0,layout_findings=findings,
        rendering_scope='Existing outcomes only; no scientific endpoint or rate selection. '
        'Any redraw bootstrap uses whole existing seeds/cells, not independent pairs.')
    if extra:provenance.update(extra)
    (REC/f'figure_{number:02d}.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (REC/f'figure_{number:02d}_caption.md').write_text(caption+'\n')
    print(path,flush=True)


def source_address_gain(ax):
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_axis_off()
    for x,w,label,color in [(.015,.20,'Readout\nloss',COLORS['ink']),
                            (.32,.29,'Neuron error\nδ₀',COLORS['bp']),
                            (.705,.27,'Dendritic\ncredit ε',COLORS['shunting'])]:
        ax.add_patch(FancyBboxPatch((x,.68),w,.245,boxstyle='round,pad=.012',
                     fc=COLORS['panel_bg'],ec=color,lw=LW_EDGE))
        ax.text(x+w/2,.80,label,ha='center',va='center',fontsize=PT_LABEL,color=color)
    for lo,hi in [(.225,.307),(.62,.698)]:
        ax.annotate('',(hi,.80),(lo,.80),arrowprops=dict(arrowstyle='->',lw=LW_DATA,color=COLORS['edge']))
    ax.text(.49,.57,'ε = A Γ c',ha='center',va='center',fontsize=PT_LABEL)
    for x,title,sub in [(.14,'A: address','which sites'),(.50,'Γ: gain','how strongly'),(.84,'c: coefficient','which signal')]:
        ax.text(x,.43,title,ha='center',fontsize=PT_ANNOT)
        ax.text(x,.31,sub,ha='center',fontsize=PT_SMALL,color=COLORS['mute'])
    ax.text(.50,.13,'Update = −learning rate × local eligibility × credit',
            ha='center',fontsize=PT_SMALL)
    ax.text(.50,.015,'Error source and spatial delivery are separate choices',
            ha='center',fontsize=PT_SMALL,color=COLORS['mute'])


def utility(ax,rows):
    variance=np.linspace(0,2,201)
    for k,q,name,color in [(1,.8,'One profile',COLORS['shunting']),
                            (2,1.,'Two profiles',COLORS['bp'])]:
        score=q*q/(q+k*variance)
        ax.plot(variance,score,label=name,color=color,lw=LW_DATA)
        rows.extend(dict(panel='C',noise_variance=float(v),rank=k,captured_energy=q,
                         twice_L_times_bound=float(y)) for v,y in zip(variance,score))
    ax.set(xlim=(0,2),ylim=(0,1.05),xticks=[0,1,2],yticks=[0,.5,1],
           xlabel='Noise variance (illustration)',ylabel='Optimized one-step bound')
    ax.legend(frameon=False,fontsize=PT_SMALL,loc='upper right',handlelength=1.25,
              handletextpad=.4,borderaxespad=.15)


def figure1():
    conditions,seeds,paired=framework.read_fresh();rows=[]
    c=NativeCanvas(493/72,3,row_weights=[135,137,110],hgutter_pt=39,vgutter_pt=45,
                   margins=Margins(left=43,right=16,top=25,bottom=39))
    source_address_gain(c.panel('A',0,0,6,schematic=True,title='From a task error to a local update',lock=False))
    framework.dictionaries(c.panel('B',0,6,6,schematic=True,title='Morphology supplies spatial profiles',lock=False))
    utility(c.panel('C',1,0,4,title='Noise and resolution',grid='y'),rows)
    d=c.panel('D',1,4,8,title='MNIST: learning with six credit rules')
    framework.accuracy(d,conditions,seeds)
    for label in d.get_legend().get_texts(): label.set_fontsize(PT_SMALL)
    framework.resolution(c.panel('E',2,0,6,title='Additional resolution within a neuron'),paired,seeds)
    legacy=framework.legacy_contrasts()
    framework.legacy_forest(c.panel('F',2,6,6,title='Separate image controls'),legacy)
    rows.extend(dict(panel='D',**r)for r in conditions.to_dict('records'))
    rows.extend(dict(panel='E',**r)for r in paired.to_dict('records'))
    rows.extend(dict(panel='F',**r)for r in legacy.to_dict('records'))
    sources=['image_ladder_controls/summaries/'+n for n in ('condition_summary_six_rules.csv','fresh_analysis_rows_six_rules.csv','paired_contrasts_six_rules.csv')]
    sources+=['mnist_between_within_factorial/paired_contrasts.csv','cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv']
    panels={'A':'Conceptual factorization, not experimental data. Exact readout errors in the image cohort; alternative source is tested separately.',
      'B':'Actual three-proximal/nine-distal projection dictionaries from build_framework.dictionaries; K1/K3 oracle projection.',
      'C':'Illustrative isotropic-noise projection bound q²/(q+Kσ²), equal smoothness L, fixed orthogonal projections; q=.8,K=1 versus q=1,K=2. Plotted value is 2L times the optimized lower bound. No endpoint-selection prediction.',
      'D':'Six-arm fresh MNIST selected-rate cohort; 10 paired seeds per architecture,180 epochs; existing confidence intervals.',
      'E':'Same-cohort paired K3−projectedK1 and exact−K3 test-accuracy differences.',
      'F':'Separate existing MNIST DFA and flattened CIFAR10 exact−neuron-specific contrasts.'}
    caption='''**Task-derived credit separates neuronal identity, spatial address and gain. A,** A readout loss supplies a neuron-specific error; a spatial dictionary A, route gains Γ and coefficients c determine the delivered field ε. A synaptic update multiplies this field by local eligibility and the negative learning rate. **B,** The actual twelve nonsomatic sites of the image model: K=1 broadcasts, K=3 groups each proximal site with its three children, and K=12 spans arbitrary site fields. Projected K=1 and K=3 use oracle coefficients and retain exact somatic errors. **C,** An explicitly illustrative one-step tradeoff: a one-dimensional projection captures q=0.8 of unit gradient energy, whereas a two-dimensional projection captures q=1. Under isotropic projected noise of variance σ², optimizing a smoothness-bound learning rate gives a bound proportional to q²/(q+Kσ²); the ordinate is twice the smoothness constant times that bound. This conditional local statement does not predict the best final trained tree. **D,** MNIST in one 128-neuron dendritic layer with a linear readout. The per-neuron condition broadcasts each neuron's exact somatic activation error. Decoder-only freezes dendritic parameters. Thin lines pair ten fresh seeds per architecture; symbols and bars show means and existing 95% intervals. Rates were selected on three separate development seeds; fits use 180 epochs and validation-selected checkpoints. **E,** Additional within-neuron resolution in the same cohort, with individual paired differences and 95% intervals. **F,** Separate historical direct-feedback-alignment (DFA) MNIST cohorts (15 seeds) and flattened CIFAR-10 (20 seeds); colors/shapes match D. pp, percentage points. Detailed one-step derivations, image fields and learning-rate controls remain in the Supplementary Information.'''
    save(c,1,sources,panels,caption,rows,{'helper_sha256':{str(Path(framework.__file__).relative_to(J)):sha(framework.__file__)}})


def task_tree(ax,center,width,kind):
    xs=np.linspace(center-width/2,center+width/2,8);previous=[(x,.15)for x in xs]
    for level,y in enumerate([.35,.55,.75]):
        current=[]
        for i in range(0,len(previous),2):
            x=(previous[i][0]+previous[i+1][0])/2
            for xx,yy in previous[i:i+2]:ax.plot([x,xx],[y,yy],color=COLORS['edge'],lw=LW_EDGE)
            symbol='×' if level==0 or(level==1 and kind=='quartet') else '+'
            ax.text(x,y,symbol,ha='center',va='center',fontsize=PT_LABEL,
                    bbox=dict(boxstyle='circle,pad=.12',fc='white',ec=COLORS['edge'],lw=LW_EDGE))
            current.append((x,y))
        previous=current
    for i,x in enumerate(xs):ax.text(x,.045,f'x{i+1}',ha='center',fontsize=PT_SMALL)
    ax.text(center,.96,TN[kind]+' target',ha='center',fontsize=PT_LABEL)


def long_curve(ax,data,task,panel,rows):
    for rule in RULES:
        p=data[data.task.eq(task)&data.optimizer.eq('adam')&data.selected_rate&data.rule.eq(rule)&data.step.ge(64)]
        w=p.pivot(index='seed',columns='step',values='test_nmse').sort_index()
        assert len(w)==20 and not w.isna().any().any()
        mean,lo,hi=boot(w.to_numpy());steps=w.columns.to_numpy()
        ax.fill_between(steps,lo,hi,color=RC[rule],alpha=.11,lw=0)
        ax.plot(steps,mean,color=RC[rule],lw=LW_DATA,label=RN[rule])
        rows.extend(dict(panel=panel,task=task,rule=rule,step=int(s),mean=float(m),ci95_low=float(l),ci95_high=float(h))for s,m,l,h in zip(steps,mean,lo,hi))
    ax.axvline(1024,color=COLORS['edge'],lw=LW_REF,ls='--')
    ax.axhline(.045 if task=='quartet'else .0225,color=COLORS['mute'],lw=LW_REF,ls=':')
    ax.set(xscale='log',yscale='log',xlim=(64,18000),ylim=(.017,2),xlabel='Training updates')
    ax.set_xticks([64,1024,16384],['64','1,024','16,384']);ax.set_yticks([.02,.1,1],['0.02','0.1','1']);ax.minorticks_off()


def figure4():
    data=read('credit_rule_extension/summaries/all_curves.csv')
    contrast=read('credit_rule_extension/summaries/paired_contrasts.csv')
    diag=read('credit_rule_extension/summaries/all_diagnostics.csv')
    diag=diag[diag.model.eq('algebraic')&diag.optimizer.eq('adam')&diag.selected_rate&diag.rule.eq('exact')&diag.state.eq('own_checkpoint')]
    original=read('credit_rule_bridge/summaries/all_diagnostics.csv')
    original=original[original.model.eq('algebraic')&original.optimizer.eq('adam')&original.selected_rate&original.rule.eq('exact')&original.state.eq('own_checkpoint')&original.step.isin([0,1024])]
    joined=original.merge(diag,on=['seed','task','step','model','optimizer','rule','rate','state'],suffixes=('_original','_extension'),validate='one_to_one')
    assert len(joined)==120
    replay_difference=float(np.max(np.abs(joined.path_best_rank_one_capture_original-joined.path_best_rank_one_capture_extension)))
    assert replay_difference<5e-16  # Round-trip CSV parsing exposes at most one floating-point ulp.
    rows=[];c=NativeCanvas(485/72,3,row_weights=[132,132,119],hgutter_pt=36,vgutter_pt=48,
                          margins=Margins(left=55,right=16,top=25,bottom=37))
    a=c.panel('A',0,0,12,title='Interaction grouping changes while the input spectrum stays fixed',schematic=True,lock=False)
    a.set(xlim=(0,1),ylim=(-.20,1.05));task_tree(a,.235,.40,'matching');task_tree(a,.765,.40,'quartet')
    a.text(.5,-.095,'Same eight inputs and tree; both targets are exactly representable',ha='center',fontsize=PT_ANNOT)
    a.text(.5,-.205,'Eight input-sensitivity eigenvalues = 1/4; signed coefficients = ±0.5',ha='center',fontsize=PT_SMALL)
    for letter,col,task in [('B',0,'matching'),('C',4,'quartet'),('D',8,'nested')]:
        ax=c.panel(letter,1,col,4,title=TN[task]+(' (separate tree)'if task=='nested'else''),grid='y')
        long_curve(ax,data,task,letter,rows)
        if letter=='B':
            ax.set_ylabel('Test NMSE');ax.legend(frameon=False,fontsize=PT_SMALL,loc='upper right',handlelength=1.4,handletextpad=.4)
        else:ax.set_yticklabels([])
    e=c.panel('E',2,0,6,title='Task-by-credit contrast over budget',grid='y')
    for endpoint,ls,marker,label in [('terminal','-','o','Last state'),('validation_selected','--','s','Validation-selected')]:
        p=contrast[contrast.task.eq('quartet_minus_matching')&contrast.optimizer.eq('adam')&contrast.metric.eq('test_nmse')&contrast.rate_view.eq('selected_rate')&contrast.endpoint.eq(endpoint)&contrast.contrast.eq('calibrated_broadcast minus exact interaction')].sort_values('budget')
        e.plot(p.budget,p['mean'],color=COLORS['bp'],ls=ls,marker=marker,ms=MARKER_MS,lw=LW_DATA,label=label)
        e.fill_between(p.budget,p.ci95_low,p.ci95_high,color=COLORS['bp'],alpha=.10,lw=0)
        rows.extend(dict(panel='E',**r)for r in p.to_dict('records'))
    e.axhline(0,color=COLORS['edge'],lw=LW_REF,ls=':');e.set_xscale('log',base=2)
    e.set(xlim=(900,18500),xlabel='Maximum training updates',ylabel='Quartic − pairwise credit deficit')
    e.set_xticks([1024,4096,16384],['1,024','4,096','16,384']);e.minorticks_off()
    e.legend(frameon=False,fontsize=PT_SMALL,loc='lower left',handlelength=1.7)
    f=c.panel('F',2,6,6,title='Credit dimension across checkpoints',grid='y')
    for task,marker,ls in [('matching','o','-'),('quartet','s','--'),('nested','^',':')]:
        results=[]
        for step in [0,1024,16384]:
            p=diag[diag.task.eq(task)&diag.step.eq(step)].sort_values('seed');assert len(p)==20 and p.seed.nunique()==20
            m,lo,hi=boot(p.path_best_rank_one_capture.to_numpy(),771009);results.append((m,lo,hi))
            rows.append(dict(panel='F',task=task,step=step,mean=m,ci95_low=lo,ci95_high=hi))
        m,lo,hi=np.array(results).T
        f.errorbar([0,1,2],m,yerr=[m-lo,hi-m],color=COLORS['ink'],marker=marker,ls=ls,lw=LW_DATA,ms=MARKER_MS,capsize=2,label=TN[task],mfc='white'if task=='quartet'else COLORS['ink'])
    f.set(xlim=(-.12,2.12),ylim=(0,1.06),xticks=[0,1,2],xticklabels=['Initial','1,024','16,384'],yticks=[0,.5,1],ylabel='Best rank-one path capture',xlabel='Training updates (checkpoint)')
    f.legend(loc='lower left',frameon=False,fontsize=PT_SMALL,handlelength=1.9,labelspacing=.2)
    sources=['credit_rule_extension/summaries/all_curves.csv','credit_rule_extension/summaries/paired_contrasts.csv','credit_rule_extension/summaries/all_diagnostics.csv','credit_rule_extension/protocol_freeze.json','credit_rule_bridge/summaries/all_diagnostics.csv','credit_rule_bridge/selection_freeze.json']
    panels={'A':'Exact pairwise/quartic constructive tree schematic, same original input spectrum. Nested task is a separate tree, no schematic equivalence implied.',
      **{p:'All20 seed blocks, three original credit rules, selected Adam rates; terminal states of the completed16384-update extension. Fixed sign control is retained in SI.'for p in ('B','C','D')},
      'E':'Existing extension paired contrasts at1024/4096/8192/16384 budgets, selected-rate Adam, terminal and validation-selected views.',
      'F':'Existing exact-rule selected-Adam own-checkpoint diagnostics at0,1024,16384. All20seeds retained at each task/checkpoint. The120initial/original1024rows agree with original bridge path capture to numerical precision;16384is the completed continuation.'}
    caption='''**Higher-order interactions require a richer credit field in these model tasks. A,** Pairwise and quartic targets share eight inputs, a tree and the same input-sensitivity eigenvalues. Multiplication and addition nodes show the grouping that makes both targets exactly representable; signs and scales are supplied by learned coefficients. **B–D,** The existing twenty seed blocks extended to 16,384 updates with the original rule-specific Adam rates and coefficient bounds. Curves show terminal-state test normalized mean-squared error (NMSE); shading gives pointwise descriptive 95% whole-seed bootstrap intervals. Dashed vertical lines mark the original 1,024-update budget; dotted horizontal lines mark task-specific noise floors. Nested targets use a separate tree. All seeds remain in the means, including slowly improving and high-error quartic fits. **E,** The paired quartic-minus-pairwise difference in initial-profile-minus-exact NMSE, evaluated at both terminal and minimum-validation-error states within each declared budget. This remains a budget-indexed comparison, not a claim that the bounded optimizers have converged. **F,** Capture of the exact nonsomatic path-credit field by its best rank-one direction at initialization, the original 1,024-update checkpoint and the existing 16,384-update continuation, with twenty-seed descriptive intervals. These are oracle field diagnostics at the exact-rule selected-Adam states. The initial and 1,024-update rows agree with the original bridge to numerical precision. Pairwise, quartic and nested means change from 0.997, 0.461 and 0.448 at 1,024 updates to 0.9999, 0.3955 and 0.4049 at 16,384. Common-rate, fixed-sign and per-seed controls are retained in the Supplementary Information.'''
    save(c,4,sources,panels,caption,rows,{'original_capture_rows_joined':len(joined),
         'original_capture_replay_max_abs_difference':replay_difference,
         'csv_parse_mode':'round_trip; replay equality checked to floating-point precision'})


def physical_curve(ax,table,arm,d,panel,rows):
    spec={('exact_autograd_bp_recipe',1):('D1 exact BP',COLORS['point_mlp'],'--'),
          ('exact_autograd_bp_recipe',3):('D3 exact BP',COLORS['bp'],'-'),
          ('path_transport',3):('D3 exact LocalCA',COLORS['pathway'],'-'),
          ('per_soma_shared',3):('D3 shared LocalCA',COLORS['local'],'-')}
    label,color,ls=spec[arm,d]
    p=table[table.arm.eq(arm)&table.depth.eq(d)&table.metric.eq('test_accuracy')].sort_values('epoch')
    ax.fill_between(p.epoch,100*p.ci95_low,100*p.ci95_high,color=color,alpha=.10,lw=0)
    ax.plot(p.epoch,100*p['mean'],color=color,ls=ls,lw=LW_DATA,label=label)
    rows.extend(dict(panel=panel,**r)for r in p.to_dict('records'))


def task_families(ax):
    """Spatial supports from the original generator, in a readable overview."""
    ax.set(xlim=(0,1),ylim=(0,1));ax.set_axis_off()
    for left,label,color,groups in [(.008,'Nested factors',COLORS['shunting'],[4,2,1]),
                                     (.345,'Flat factors',COLORS['additive'],[8,8,8])]:
        ax.add_patch(FancyBboxPatch((left,.02),.305,.93,boxstyle='round,pad=.006',
                     facecolor=COLORS['panel_bg'],edgecolor=color,lw=LW_HAIR))
        ax.text(left+.1525,.98,label,ha='center',va='top',fontsize=PT_LABEL,color=color,
                bbox=dict(fc='white',ec='none',pad=.4))
        for level,(y,n)in enumerate(zip([.72,.55,.38],groups),1):
            ax.text(left+.028,y,f'h{level}',ha='right',va='center',fontsize=PT_SMALL)
            for k in range(n):
                width=.257/n
                ax.add_patch(FancyBboxPatch((left+.039+k*width,y-.050),width-.003,.1,
                             boxstyle='round,pad=0',fc=color,ec=color,lw=LW_HAIR,alpha=.27))
        ax.text(left+.1525,.17,'Class signal × nuisance gains',ha='center',fontsize=PT_SMALL)
        ax.text(left+.1525,.045,'Fine · coarse · global'if groups==[4,2,1]else'Equal-resolution gain groups',ha='center',fontsize=PT_SMALL,color=COLORS['mute'])
    left=.685;color=COLORS['highlight']
    ax.add_patch(FancyBboxPatch((left,.02),.305,.93,boxstyle='round,pad=.006',
                 facecolor=COLORS['panel_bg'],edgecolor=color,lw=LW_HAIR))
    ax.text(left+.1525,.98,'Local ratios',ha='center',va='top',fontsize=PT_LABEL,color=color,
            bbox=dict(fc='white',ec='none',pad=.4))
    for k,x in enumerate([.745,.84,.935],1):
        for dx,letter,col in [(-.018,'E',COLORS['exc']),(.018,'I',COLORS['inh'])]:
            ax.text(x+dx,.76,letter,ha='center',fontsize=PT_SMALL,color=col)
            ax.scatter(x+dx,.65,s=19,color=col,zorder=3)
        ax.annotate('',(x,.43),(x,.60),arrowprops=dict(arrowstyle='->',color=color,lw=LW_EDGE))
        ax.text(x,.38,f'R{k}',ha='center',va='center',fontsize=PT_LABEL,
                bbox=dict(boxstyle='round,pad=.3',fc='white',ec=color,lw=LW_EDGE))
    ax.text(left+.1525,.17,'Division within each module',ha='center',fontsize=PT_SMALL)
    ax.text(left+.1525,.045,'Class signal remains distal',ha='center',fontsize=PT_SMALL,color=COLORS['mute'])


def figure6():
    rows=[];c=NativeCanvas(535/72,4,row_weights=[113,113,110,112],hgutter_pt=36,vgutter_pt=40,
                          margins=Margins(left=52,right=17,top=25,bottom=35))
    a=c.panel('A',0,0,12,title='Task structure sets the spatial organization of nuisance gains',schematic=True,lock=False)
    task_families(a)
    b=c.panel('B',1,0,6,title='Task hierarchy and physical depth')
    h4=read('physical_depth_h4_factorial/seed_outcomes.csv');matrix=depth.hierarchy_matrix(h4)
    grid=pd.concat([read('physical_depth_clean_source_replication/seed_outcomes.csv'),h4],ignore_index=True)
    grid=grid[grid.hierarchy.isin([2,3,4])&grid.regime.eq('aligned')&grid.architecture.eq('serial_tree')&grid.mechanism.eq('shunting')&grid.credit.eq('full_bp')]
    counts=grid.groupby(['hierarchy','depth']).size()
    assert counts.eq(10).all() and len(counts)==9
    depth.heatmap(b,matrix,['H = 2','H = 3','H = 4'],['D1','D2','D3','D4'],best_by_row=True)
    for i,j in np.argwhere(~np.isfinite(matrix)):b.text(j,i,'—',ha='center',va='center',fontsize=PT_SMALL)
    b.set_xlabel('Physical depth (180-epoch budget)');b.set_ylabel('Task gain tiers')
    for i,j in np.argwhere(np.isfinite(matrix)):rows.append(dict(panel='B',hierarchy=i+2,depth=j+1,mean_test_accuracy=float(matrix[i,j]),n_seeds=int(counts.loc[i+2,j+1])))
    cc=c.panel('C',1,6,6,title='Serial computation depends on the task',grid='y')
    effects=read('task_family_alignment/architecture_effects.csv')
    depth.panel_alignment(cc,effects,'bp',left=True)
    cc.set_xlabel('Task–sensor alignment (180 epochs)')
    rows.extend(dict(panel='C',**r)for r in effects[effects.credit.eq('bp')].to_dict('records'))
    d=c.panel('D',2,0,12,title='The nested-task depth benefit persists during longer training',grid='y')
    curves=read('physical_depth_followup/condition_trajectory_summary.csv')
    for arm,dd in [('exact_autograd_bp_recipe',1),('exact_autograd_bp_recipe',3),('path_transport',3),('per_soma_shared',3)]:physical_curve(d,curves,arm,dd,'D',rows)
    d.set(xlim=(0,610),ylim=(32,102),xticks=[0,180,400,600],xlabel='Epoch',ylabel='Test accuracy (%)')
    d.axvline(180,color=COLORS['mute'],lw=LW_REF,ls=':')
    d.legend(frameon=False,fontsize=PT_SMALL,ncol=2,loc='lower right',handlelength=1.6,labelspacing=.1)
    gaps=read('physical_depth_followup/paired_trajectory_summary.csv')
    for letter,col,metric,title,ylabel in [('E',0,'test_accuracy','Accuracy changes its ordering','Exact − shared accuracy (pp)'),('F',6,'test_cross_entropy','Cross-entropy retains its ordering','Exact − shared cross-entropy')]:
        ax=c.panel(letter,3,col,6,title=title,grid='y');p=gaps[gaps.metric.eq(metric)].sort_values('epoch')
        ax.fill_between(p.epoch,p.ci95_low,p.ci95_high,color=COLORS['pathway'],alpha=.13,lw=0)
        ax.plot(p.epoch,p['mean'],color=COLORS['pathway'],lw=LW_DATA)
        ax.axhline(0,color=COLORS['edge'],lw=LW_REF,ls='--');ax.axvline(180,color=COLORS['mute'],lw=LW_REF,ls=':')
        ax.set(xlim=(0,615),xticks=[0,180,400,600],xlabel='Epoch',ylabel=ylabel)
        rows.extend(dict(panel=letter,**r)for r in p.to_dict('records'))
        if letter=='E':
            for epoch,label,offset in [(180,'+10.86',(12,0)),(600,'−1.52',(-30,15))]:
                z=p[p.epoch.eq(epoch)].iloc[0];ax.scatter(epoch,z['mean'],s=20,color=COLORS['pathway']);ax.annotate(label,(epoch,z['mean']),xytext=offset,textcoords='offset points',fontsize=PT_SMALL)
        else:ax.text(.04,.06,'Negative = lower loss with exact credit',transform=ax.transAxes,fontsize=PT_SMALL)
    sources=['physical_depth_h4_factorial/seed_outcomes.csv','physical_depth_clean_source_replication/seed_outcomes.csv',
             'task_family_alignment/architecture_effects.csv','physical_depth_followup/condition_trajectory_summary.csv',
             'physical_depth_followup/paired_trajectory_summary.csv','physical_depth_followup/analysis_validation.json']
    panels={'A':'Restored original main6A/S31A native task-family schematic. Same commutative product in nested/flat families; spatial factor supports differ.',
      'B':'Restored S31E, exact-BP180-epoch means from clean-source H2/H3 and H4 cohorts; no outcome selection beyond declared conditions. H2D3/H2D4/H3D4 unobserved and blank.',
      'C':'Restored S31F, paired serial−grouped point BP task-family alignment effects at180epochs. Resource-identical modules/contacts; all10paired seeds.',
      'D':'Existing followup validation-selected test-accuracy trajectories,10seeds/condition. D1[8],D3[2,1,2];8compartments,64somata,matched contacts.',
      'E':'Paired exact−shared LocalCA test-accuracy trajectory with archived pointwise intervals.',
      'F':'Same paired states, test-cross-entropy difference; negative favors exact credit.'}
    caption='''**Task organization determines when serial dendritic computation helps. A,** Nested and flat-factor tasks share a commutative product of a distal class signal and nuisance gains. Nested factors have fine, coarse and global supports; flat factors have equal-resolution supports. Local-ratio tasks expose the relevant division within each module. The drawings describe input access, not exact cancellation. **B,** Mean test accuracy for aligned serial trees with two to four task gain tiers H and tested physical depths D, using exact BP at the original 180-epoch budget. Each entry averages the original seed cohort; outlines mark the largest tested mean in each row. H=2 favors D2, while H=3 and H=4 favor D3 among the tested choices. Only D1–D2 were tested for H=2, D1–D3 for H=3 and D1–D4 for H=4. These are fixed-budget comparisons, not converged optimal-depth estimates. **C,** Paired accuracy advantage of serial over resource-identical grouped-point computation under exact BP, across task families and sensor alignment; means and retained 95% intervals from ten pairs at 180 epochs. **D,** Longer nested-task trajectories for D1 [8] and D3 [2,1,2], with eight nonsomatic compartments per neuron, 64 somata and matched resources. Curves evaluate validation-selected states, carrying stopped states forward. **E,F,** Paired D3 exact-path-minus-shared-soma LocalCA differences in test accuracy and cross-entropy. The accuracy contrast changes from +10.86 percentage points at 180 epochs to −1.52 at 600, whereas exact credit retains lower mean cross-entropy. All fifty D3 fits reach the 600-epoch cap; eight of ten D1 references stop earlier. Shading in D–F shows archived pointwise descriptive 95% whole-seed intervals. The two rules are compared at specified budgets and metrics; neither is established as the converged winner. Architecture, optimizer and reversed-placement controls remain in the Supplementary Information.'''
    save(c,6,sources,panels,caption,rows,{'helper_sha256':{str(Path(depth.__file__).relative_to(J)):sha(depth.__file__)}})


def dictionary_cartoon(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.set_axis_off()
    pos={0:(.18,.86),1:(.08,.61),2:(.28,.61),3:(.025,.35),4:(.13,.35),5:(.23,.35),6:(.335,.35)}
    for i in range(1,7):
        x,y=pos[i];xx,yy=pos[(i-1)//2]
        ax.plot([x,xx],[y,yy],color=COLORS['shunting']if i in(1,3,4)else COLORS['mute'],lw=LW_DATA)
    for i,(x,y)in pos.items():ax.scatter(x,y,s=25,color=COLORS['shunting']if i in(1,3,4)else COLORS['mute'],zorder=3)
    ax.text(.18,.17,'Illustrative arbor',ha='center',fontsize=PT_SMALL)
    matrix=np.column_stack([np.ones(6),[1,0,1,1,0,0],[0,1,0,0,1,1]])
    inner=ax.inset_axes([.51,.28,.35,.59]);inner.imshow(matrix,aspect='auto',cmap=ListedColormap(['white',COLORS['shunting']]),vmin=0,vmax=1)
    inner.set_xticks([0,1,2],['Common','Left','Right']);inner.set_yticks([]);inner.tick_params(length=0,labelsize=PT_SMALL)
    inner.set_ylabel('Nonsomatic sites',fontsize=PT_LABEL)
    ax.text(.5,.03,'One common signal + ancestry-defined spatial profiles',ha='center',fontsize=PT_SMALL)


def figure7():
    tables={q:read('anatomy_commonmode/'+q+'/cell_method_summary.csv')for q in ('original8','v661','pinky')}
    table=tables['v661'];report=json.loads((S/'anatomy_commonmode/v661/summary.json').read_text())
    rows=[];c=NativeCanvas(493/72,3,row_weights=[137,133,127],hgutter_pt=36,vgutter_pt=45,
                          margins=Margins(left=48,right=20,top=26,bottom=36))
    anatomy.panel_arbor(c.panel('A',0,0,6,title='Measured arbor and mapped contacts',schematic=True,lock=False))
    dictionary_cartoon(c.panel('B',0,6,6,title='The dictionary includes a shared signal',schematic=True,lock=False))
    cc=c.panel('C',1,0,6,title='Ancestry capacity across budgets',grid='y')
    specs=[(commonmode.METHODS[0],COLORS['shunting'],'o','Ancestry'),(commonmode.METHODS[1],COLORS['mute'],'s','Surrogate tree'),(commonmode.METHODS[2],COLORS['additive'],'^','Depth bins'),(commonmode.METHODS[-1],COLORS['ink'],'D','SVD oracle')]
    for method,color,marker,label in specs:
        p=table[table.method.eq(method)&table.channels.isin([1,2,4,8])].pivot(index='root_id',columns='channels',values='residual_capture').sort_index();assert p.shape==(47,4) and not p.isna().any().any()
        m,lo,hi=boot(p.to_numpy(),2609087);k=p.columns.to_numpy()
        cc.plot(k,m,color=color,marker=marker,ms=3,lw=LW_DATA,label=label);cc.fill_between(k,lo,hi,color=color,alpha=.075,lw=0)
        rows.extend(dict(panel='C',method=method,channels=int(kk),mean=float(mm),ci95_low=float(ll),ci95_high=float(hh))for kk,mm,ll,hh in zip(k,m,lo,hi))
    cc.set_xscale('log',base=2);cc.set(xlim=(.9,8.8),ylim=(-.02,1.02),xlabel='Profiles K (including the common signal)',ylabel='Spatial-residual energy captured')
    cc.set_xticks([1,2,4,8],['1','2','4','8']);cc.minorticks_off();cc.legend(frameon=False,fontsize=PT_SMALL,loc='upper left',handlelength=1.3,labelspacing=.15)
    d=c.panel('D',1,6,6,title='The spatial controls at K = 8')
    contrasts=commonmode.contrast_forest(d,report);d.set_xlabel('Ancestry advantage in residual capture (pp)');d.tick_params(axis='y',labelsize=PT_SMALL)
    rows.extend(dict(panel='D',**r)for r in contrasts.to_dict('records'))
    e=c.panel('E',2,0,6,title='Capture and wiring at K = 8',grid='both')
    focus=table[table.channels.eq(8)].groupby('method').mean(numeric_only=True)
    label_positions={'Ancestry':(31,.64),'Surrogate tree':(40,.51),'Depth bins':(44,.41),
                     'Random routes':(31,.15),'Shuffled routes':(43,.29),'SVD oracle':(99,.79)}
    for method,label in zip(commonmode.METHODS,commonmode.LABELS):
        r=focus.loc[method];color=COLORS['shunting']if method==commonmode.METHODS[0]else COLORS['additive']if label=='Depth bins'else COLORS['ink']if label=='SVD oracle'else COLORS['mute']
        e.scatter(100*r.wiring_density,r.residual_capture,s=25,color=color,zorder=3)
        x,y=label_positions[label]
        e.annotate(label,(100*r.wiring_density,r.residual_capture),xytext=(x,y),
                   textcoords='data',fontsize=PT_SMALL,color=color,
                   ha='right'if label=='SVD oracle'else'left',va='center',
                   arrowprops=dict(arrowstyle='-',color=color,lw=LW_HAIR,shrinkA=2,shrinkB=3))
        rows.append(dict(panel='E',method=method,**r.to_dict()))
    e.set(xlim=(0,106),ylim=(0,1.05),xlabel='Nonzero coefficients (% of dense wiring)',ylabel='Spatial-residual energy captured')
    f=c.panel('F',2,6,6,title='Capacity across three cohorts')
    cohorts=commonmode.cohort_points(f,tables);rows.extend(dict(panel='F',**r)for r in cohorts.to_dict('records'))
    sources=['figure3/segment_metrics.csv','anatomy_commonmode/protocol_freeze.json']
    sources +=['anatomy_commonmode/'+q+'/'+n for q in tables for n in ('cell_method_summary.csv','summary.json')]
    panels={'A':'Original median-sized example actual PCA geometry and mapped E/I contacts, unchanged helper.',
      'B':'Explicitly illustrative ancestry dictionary with constant column; not an inferred measured-support matrix.',
      'C':'Corrected common-mode residual-capture atK1/2/4/8, the complete47-cell budgets, whole-cell descriptive bootstrap. K16is available only in a cell subset and is not displayed here; original source rows remain untouched.',
      'D':'Archived paired47-cell K8ancestry residual-capture differences against allfour spatial controls; surrogate first.',
      'E':'Same47cells,K8:mean residual capture versus actual nonzero density; no claim equalK fixes rank or wiring.',
      'F':'Original8,disjoint47 and second-mouse8 common+ancestry total/residual capture, archived helper fixed-seed cell bootstrap.'}
    caption='''**Anatomy supplies sparse spatial dictionaries beyond a shared broadcast. A,** A reconstructed arbor from the original eight-cell cohort, chosen by median segment count. Segment color reflects mapped excitatory/inhibitory contact area; line width encodes total contact area, and the scale bar is 50 μm. **B,** An explicitly illustrative dictionary combines a constant profile with ancestry-defined subtree profiles. **C,** Capture of the modeled field remaining after the weighted common projection, across the four complete profile budgets K=1,2,4,8 in 47 disjoint v661 cells. Every dictionary contains the same constant profile. Lines show cell means; shading gives descriptive whole-cell 95% intervals. The common-constrained SVD oracle is a representational ceiling. **D,** At K=8, paired ancestry advantages over surrogate-tree, depth-bin, random-site and shuffled-route controls, with the retained 95% cell-bootstrap intervals. Surrogates preserve segment depth and parent out-degree, providing the closest topology control. **E,** The same K=8 dictionaries compared by average residual capture and nonzero coefficient density relative to dense eight-column wiring. Equal K does not equate rank or wiring; ancestry and its shuffled control have the same nonzero counts. **F,** Common-plus-ancestry total and residual capture across the original eight cells, 47 disjoint cells from the same mouse and eight eligible Pinky cells from a second mouse. Pinky's near-saturation involves only 9–13 excitatory-bearing sites per cell. Fields are modeled responses to focal passive shunts and therefore carry ancestry structure through cable physics; these comparisons establish representational capacity, not independent evidence of endogenous route use.'''
    save(c,7,sources,panels,caption,rows,{'helper_sha256':{str(Path(p.__file__).relative_to(J)):sha(p.__file__)for p in (anatomy,commonmode)}})


def figure9():
    rows=[];c=NativeCanvas(365/72,2,row_weights=[135,153],hgutter_pt=36,vgutter_pt=52,
                          margins=Margins(left=58,right=17,top=26,bottom=39))
    a=c.panel('A',0,0,12,title='Observed ancestry–response similarity in seven targets')
    sums=read('review_evidence_reanalysis/functional_native_contrasts.csv');vals=read('review_evidence_reanalysis/functional_native_target_effects.csv');data=[]
    for mode,label in [('selected_scans','Selected scans'),('scan_complete','All eligible scans')]:
        r=sums[sums.endpoint.eq('structure_function_partial_r')&sums.comparison.eq(mode)].iloc[0]
        v=vals[vals.endpoint.eq('structure_function_partial_r')&vals.comparison.eq(mode)].effect.to_numpy();assert len(v)==7
        data.append((label,v,(r['mean'],r.ci95_low,r.ci95_high),COLORS['shunting']))
        rows.append(dict(panel='A',**r.to_dict()))
    measured.forest(a,data,'Partial rank correlation',(-.65,.55));a.axvline(0,color=COLORS['mute'],lw=LW_REF,ls='--')
    b=c.panel('B',1,0,6,title='Sensitivity with the measured sampling',grid='y')
    power=read('measured_alignment_power/power_summary.csv')
    for scenario,color,ls,label in [('measured',COLORS['shunting'],'-','Measured reliability'),('perfect',COLORS['mute'],'--','Perfect reliability')]:
        p=power[power.reliability.eq(scenario)].sort_values('lambda')
        b.plot(p['lambda'],p.power,color=color,ls=ls,lw=LW_DATA,label=label)
        b.fill_between(p['lambda'],p.power_ci95_low,p.power_ci95_high,color=color,alpha=.12,lw=0)
        rows.extend(dict(panel='B',**r)for r in p.to_dict('records'))
    b.axhline(.8,color=COLORS['edge'],lw=LW_REF,ls=':');b.axvspan(.55,.60,ymin=.72,ymax=.82,color=COLORS['shunting'],alpha=.15,lw=0)
    b.set(xlim=(0,1),ylim=(0,1.05),xticks=[0,.5,1],yticks=[0,.5,.8,1],xlabel='Ancestry variance fraction λ (simulated)',ylabel='Positive-alignment detection probability')
    b.legend(frameon=False,fontsize=PT_SMALL,loc='lower right',handlelength=1.5)
    cc=c.panel('C',1,6,6,title='Repeat reliability of the observed inputs',grid='y')
    reliability=read('measured_alignment_power/reliability_calibration_audit.csv')
    values=reliability.measured_split_half_spearman.to_numpy();assert len(values)==125
    bins=np.linspace(-.3,1,14);assert values.min()>=bins[0]and values.max()<=bins[-1]
    cc.hist(values,bins=bins,color=COLORS['mute'],edgecolor='white',lw=LW_HAIR)
    cc.axvline(0,color=COLORS['edge'],lw=LW_REF,ls=':')
    cc.set(xlim=(-.3,1),xticks=[0,.5,1],xlabel='Split-half Spearman correlation',ylabel='Partner–scan observations')
    cc.text(.97,.97,'125 observations\n102 partners · 13 scans',ha='right',va='top',transform=cc.transAxes,fontsize=PT_SMALL)
    rows.extend(dict(panel='C',**r)for r in reliability.to_dict('records'))
    sources=['review_evidence_reanalysis/functional_native_contrasts.csv','review_evidence_reanalysis/functional_native_target_effects.csv',
             'measured_alignment_power/power_summary.csv','measured_alignment_power/reliability_calibration_audit.csv',
             'measured_alignment_power/RESULTS.json','measured_alignment_power/protocol_freeze.json']
    panels={'A':'Original empirical7target partial-rank estimates; selected/all-eligible scan policies, originalCI.',
      'B':'Completed4000joint-dataset simulation; observed partners/shared-pair dependencies,13scans7targets,exact two-sided n7signrank plus positive effect; measured/perfect reliability.',
      'C':'Measured repeat Spearman reliabilities for all125partner-recording observations (102unique partners); no simulation outcomes in histogram. No independent-observation test.'}
    caption='''**Measured responses provide no evidence of preferential ancestry alignment at the available sensitivity. A,** Empirical ancestry–response partial-rank correlations after distance adjustment. Small points are seven target-level estimates; diamonds and bars show means and the archived descriptive 95% target-bootstrap intervals. Selected scans and all eligible scans are displayed separately. **B,** Conditional sensitivity simulations preserve the actual partners, repeated partners and shared-pair dependence across thirteen scans in seven targets. The horizontal axis λ is the simulated fraction of latent response variance allocated to the specified Gaussian ancestry kernel; it is not an observed correlation. Each point summarizes 4,000 complete simulated datasets tested by the exact two-sided seven-unit signed-rank test at 0.05, additionally requiring a positive mean effect. Shading is a 95% Monte Carlo interval. With measured repeat attenuation, sustained 80% detection is bracketed by λ=0.55–0.60; the corresponding interpolated mean partial-rank effect is approximately 0.249 within this model family. This is not a universal detectable biological correlation. The exact seven-unit test has minimum two-sided P=0.015625; no universal power ceiling below one follows from that lattice. **C,** The measured split-half response correlations used for attenuation, retaining all 125 partner–scan records from 102 unique partners and thirteen scans. Negative measured reliabilities are displayed here and clipped to zero only in the simulation's nonnegative signal-variance calibration. The measured-reliability null gives two-sided type-I error 0.0495. Actual route support, coverage, response-prediction comparisons with ridge and fixed-profile reconstruction remain in the Supplementary Information.'''
    save(c,9,sources,panels,caption,rows,{'helper_sha256':{str(Path(measured.__file__).relative_to(J)):sha(measured.__file__)}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',type=int,default=[1,4,6,7,9],choices=[1,4,6,7,9])
    parser.add_argument('--emit-main',action='store_true')
    args=parser.parse_args()
    for number in args.figures:
        globals()[f'figure{number}']()
        if args.emit_main:shutil.copyfile(OUT/f'restored_main_{number:02d}.pdf',J/f'figures/main/figure_{number:02d}.pdf')


if __name__=='__main__':main()
