#!/usr/bin/env python3
"""Proposed main Figure 5: physical credit and the longer-budget control.

Original canonical figures remain unchanged. Requires all 60 extended outcomes.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from extend import OUT,JOURNAL,dump,sha
sys.path.insert(0,str(JOURNAL/'scripts'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from journal_style import (apply_neurips_style, COLORS, FIG_W, LW_DATA, LW_ERR, LW_REF, PT_SMALL, PT_ANNOT, PT_LEGEND, PT_TITLE, PANEL_LABEL_PT, MARKER_MS, SEED_MS, SEED_ALPHA, ERR_CAPSIZE)

SPECS=[('full_bp','Exact; BP recipe','bp','o','-'),
 ('soma_broadcast_bp','Broadcast; BP recipe','additive','s','--'),
 ('soma_broadcast_matched','Broadcast; LocalCA recipe','highlight','v','--'),
 ('local_path','Exact path; LocalCA','pathway','^','-.'),
 ('local_shared','Shared soma; LocalCA','local','D',':')]
MAP={'exact_autograd_bp_recipe':('bp','Exact; BP recipe'),
 'broadcast_autograd_bp_recipe':('additive','Broadcast; BP recipe'),
 'broadcast_autograd_localca_recipe':('highlight','Broadcast; LocalCA recipe'),
 'path_transport':('pathway','Exact path; LocalCA'),'per_soma_shared':('local','Shared soma; LocalCA')}


def main():
    data=OUT/'canonical'
    audit=json.loads((data/'extension_validation.json').read_text());assert audit['status']=='passed'
    summary=pd.read_csv(JOURNAL/'source_data/point_dendrite_credit_controls/condition_summary.csv')
    curves=pd.read_csv(data/'extension_validation_trajectories.csv');contrasts=pd.read_csv(data/'extension_paired_contrasts.csv');seeds=pd.read_csv(data/'extension_paired_seed_contrasts.csv')
    apply_neurips_style();fig,axes=plt.subplots(3,1,figsize=(FIG_W,7.8));fig.subplots_adjust(left=.12,right=.98,bottom=.08,top=.96,hspace=.58)
    ax=axes[0]
    for credit,label,color,marker,ls in SPECS:
        part=summary[summary.architecture.eq('serial_tree')&summary.regime.eq('aligned')&summary.credit.eq(credit)].sort_values('depth')
        means=part.mean_test_accuracy.to_numpy()*100
        ax.errorbar(part.depth,means,yerr=[means-100*part.ci95_low_test_accuracy,100*part.ci95_high_test_accuracy-means],
            color=COLORS[color],marker=marker,ls=ls,ms=MARKER_MS,capsize=ERR_CAPSIZE,lw=LW_DATA,elinewidth=LW_ERR,label=label)
    ax.set_xticks([1,2,3],['D1','D2','D3']);ax.set_xlim(.8,3.2);ax.set_ylim(44,105)
    ax.set_ylabel('Test accuracy (%)');ax.set_xlabel('Physical depth')
    ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=PT_LEGEND,columnspacing=1.7)
    ax.set_title('Original 180-epoch comparison',loc='left',pad=8)
    ax=axes[1]
    for (arm,depth),part in curves.groupby(['arm','depth']):
        mean=part.groupby('epoch').best_validation_loss.mean()
        key,label=MAP[arm];color=COLORS[key];ls='-'
        if depth==1:color=COLORS['point_mlp'];ls='--';label='D1 exact; BP recipe'
        ax.plot(mean.index,mean,color=color,ls=ls,label=label)
    ax.axvline(180,color=COLORS['mute'],ls=':',lw=LW_REF);ax.set_xlim(0,600)
    ax.set_xlabel('Epoch');ax.set_ylabel('Best validation loss')
    ax.set_title('Longer budget, with original stopping rule',loc='left',pad=8)
    ax.text(184,.97,'180',transform=ax.get_xaxis_transform(),va='top',fontsize=PT_LEGEND,color=COLORS['mute'])
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.26),ncol=3,frameon=False,fontsize=PT_SMALL,columnspacing=1.5)
    ax=axes[2]
    labels=[('depth_gain_exact_bp','Depth: exact BP, D3−D1'),
        ('localca_path_minus_shared','LocalCA: exact path−shared soma'),
        ('bp_exact_minus_broadcast','BP recipe: exact−broadcast'),
        ('broadcast_bp_minus_localca_recipe','Broadcast: BP−LocalCA recipe')]
    for index,(key,label) in enumerate(labels):
        for budget,offset,color,marker in [(180,-.12,'point_mlp','o'),(600,.12,'additive','s')]:
            row=contrasts[contrasts.contrast.eq(key)&contrasts.budget.eq(budget)].iloc[0]
            values=seeds[seeds.contrast.eq(key)&seeds.budget.eq(budget)].sort_values('seed').difference_pp.to_numpy()
            assert len(values)==10
            ax.scatter(values,index+offset+np.linspace(-.045,.045,10),s=SEED_MS**2,color=COLORS[color],alpha=SEED_ALPHA*.65,zorder=2)
            ax.errorbar(row['mean'],index+offset,xerr=[[row['mean']-row.ci95_low],[row.ci95_high-row['mean']]],
                fmt=marker,ms=MARKER_MS,capsize=ERR_CAPSIZE,color=COLORS[color],lw=LW_ERR,label=f'{budget}-epoch budget' if index==0 else None)
    ax.set_yticks(range(4),[v[1] for v in labels],fontsize=PT_ANNOT);ax.invert_yaxis()
    ax.axvline(0,color=COLORS['mute'],ls=':',lw=LW_REF);ax.set_xlabel('Paired test-accuracy difference (percentage points)')
    fig.text(.12,ax.get_position().y1+.014,'Credit contrasts in the same extended trajectories',ha='left',fontsize=PT_TITLE)
    ax.legend(loc='lower right',frameon=False,fontsize=PT_LEGEND)
    # Long contrast descriptions use a separate left label column at full figure width.
    position=ax.get_position();ax.set_position([.48,position.y0,.50,position.height])
    for letter,ax in zip('ABC',axes):
        ax.spines[['top','right']].set_visible(False)
        position=ax.get_position();fig.text(.025,position.y1+.008,letter,fontsize=PANEL_LABEL_PT,fontweight='bold')
    dest=OUT/'figure';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'figure5_physical_depth_budget.pdf');fig.savefig(dest/'figure5_physical_depth_budget.png',dpi=160);plt.close(fig)
    import fitz
    document=fitz.open(dest/'figure5_physical_depth_budget.pdf');page=document[0]
    spans=[span for block in page.get_text('dict')['blocks'] if 'lines' in block for line in block['lines'] for span in line['spans'] if span['text'].strip()]
    outside=[span['text'] for span in spans if not page.rect.contains(fitz.Rect(span['bbox']))]
    dump(dest/'native_design_audit.json',dict(width_pt=page.rect.width,height_pt=page.rect.height,min_font_pt=min(span['size'] for span in spans),max_font_pt=max(span['size'] for span in spans),outside_text=outside,shared_panel_label_pt=PANEL_LABEL_PT))
    assert abs(page.rect.width-FIG_W*72)<.02
    assert not outside
    caption='Physical depth and credit delivery in the aligned hierarchical gain task. A, Original 180-epoch depth comparison with fixed branch resources and five delivery/optimizer conditions; points show means and 95% seed-bootstrap intervals. B, Best validation loss during the 60 new source-consistent restarts with maximum 600 epochs and the original patience 30; stopped runs retain their last best loss, so each curve includes the same 10 seeds throughout. All colored curves are D3; the gray dashed curve is the D1 exact-BP reference. The vertical line marks 180 epochs. C, Paired test-accuracy contrasts from validation-selected weights within the 180- and 600-epoch windows of these same trajectories. Faint points are the ten paired seed differences; larger symbols show paired means and descriptive 95% whole-seed bootstrap intervals. Both optimizer recipes use Adam and retain their original parameter groups, learning rates and update options. Every outcome is retained; an extended finite budget is not a convergence guarantee.'
    (dest/'caption.txt').write_text(caption+'\n')
    dump(dest/'provenance.json',dict(builder_sha256=sha(__file__),inputs={str(p.relative_to(JOURNAL)):sha(p) for p in [JOURNAL/'source_data/point_dendrite_credit_controls/condition_summary.csv',data/'extension_validation_trajectories.csv',data/'extension_paired_contrasts.csv',data/'extension_paired_seed_contrasts.csv']},numerical_changes=False,new_fits=60))


if __name__=='__main__':main()
