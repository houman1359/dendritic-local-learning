#!/usr/bin/env python3
"""Original prospective morphology selection, retained as Supplementary Figure S35.

The assembly map assigns the final publication number (Figure 6). Every
outcome comes from the sealed prospective experiment; no training or model
selection occurs in this renderer. Strong baselines and negative results
receive the same visual prominence as the selector.

2026-09-11 visual-review pass (S34 work list): the two arm-wise regret panels
are merged into one four-row, two-arm dot plot (C) keyed once; the seed
clouds are jittered across the arm band with an open mean marker on top; the
regret axis stops at 0.15; the budget panel (D) draws the twenty seed-block
means per rank, integer ticks 1-8 and one gray reference line for the
retrospectively best budget (K = rank in every case, both arms) instead of two
coincident dashed curves; the horizon panel (E) draws the seed-block means and
clips its axis to the data; the candidate schematic (A) draws the four
balanced leaf assignments and the comb from the manifest edges with numbered
leaves in neutral gray and shows the K = 1, 2, 4, 8 cuts on one tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_HAIR, LW_REF,
                           PT_ANNOT, PT_LABEL, PT_SMALL, Margins, NativeCanvas)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"source_data/prospective_morphology_selection"
OUT=ROOT/"figures/supplementary/figure_S35_panels_A-F.pdf"
ARMS=["feedback_only","joint_forward_feedback"]
ARM_COLORS=[COLORS["shunting"],COLORS["additive"]]
ARM_LABELS={"feedback_only":"feedback only","joint_forward_feedback":"joint transfer"}
POLICIES=["moment_selector","rank_only","development_best","uniform_random_expectation"]
LABELS={"moment_selector":"moment selector","rank_only":"rank-only",
        "development_best":"fixed / max-budget","uniform_random_expectation":"random expectation"}
TREES=[("balanced_contiguous","balanced 1"),("balanced_interleaved","balanced 2"),
       ("balanced_permutation_a","balanced 3"),("balanced_permutation_b","balanced 4"),
       ("comb_contiguous","comb")]
BUDGETS=[1,2,4,8]
JITTER_SEED=20260911   # display jitter only; never touches a plotted value


def _tree(item):
    """Children map, root and the in-order leaf sequence of one manifest tree."""
    children={}
    parent={}
    for a,b,_ in item["edges"]:
        children.setdefault(int(a),[]).append(int(b));parent[int(b)]=int(a)
    root=next(n for n in children if n not in parent)
    order=[]   # the manifest lists children in slot order (left to right)
    def walk(n):
        if n<8:order.append(n)
        for c in children.get(n,[]):walk(c)
    walk(root)
    return children,root,order


def _layout(children,root,order,x0,width,y_leaf,y_root):
    """Slot leaves left to right, internal nodes at the midpoint of their leaves."""
    depth={};
    def d(n,k):
        depth[n]=k
        for c in children.get(n,[]):d(c,k+1)
    d(root,0)
    maxd=max(depth.values())
    pts={}
    slot={leaf:i for i,leaf in enumerate(order)}
    def place(n):
        if n<8:
            pts[n]=np.array([x0+width*slot[n]/7,y_leaf]);return [slot[n]]
        leaves=[]
        for c in children[n]:leaves+=place(c)
        pts[n]=np.array([x0+width*np.mean(leaves)/7,
                         y_leaf+(y_root-y_leaf)*(maxd-depth[n])/maxd])
        return leaves
    place(root)
    return pts


def _cut_nodes(item,children,root):
    """Subtree roots of the K route channels, from the manifest dictionary."""
    leaves_of={}
    def collect(n):
        leaves_of[n]=[n] if n<8 else sum((collect(c) for c in children[n]),[])
        return leaves_of[n]
    collect(root)
    dictionary=np.asarray(item["dictionary"],float)
    nodes=[]
    for col in range(dictionary.shape[1]):
        members=sorted(int(i) for i in np.flatnonzero(np.abs(dictionary[:,col])>0))
        nodes.append(next(n for n in leaves_of if sorted(leaves_of[n])==members))
    return nodes


def candidate_panel(ax,manifest):
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ink=COLORS["mute"];edge=COLORS["dend"]
    by_id={c["candidate_id"]:c for c in manifest}
    slot_w=.186;gap=(1-5*slot_w)/4
    # top strip: the five labelled trees, leaves numbered by input index
    for i,(tree_id,label) in enumerate(TREES):
        item=by_id[f"{tree_id}_k4"]
        children,root,order=_tree(item)
        assert order==[int(v) for v in item["permutation"]],tree_id
        x0=i*(slot_w+gap)+.012;w=slot_w-.024
        pts=_layout(children,root,order,x0,w,.66,.97)
        for n,kids in children.items():
            for c in kids:
                ax.plot([pts[n][0],pts[c][0]],[pts[n][1],pts[c][1]],color=edge,lw=LW_EDGE,
                        solid_capstyle="round",zorder=2)
        for leaf in order:
            ax.plot(*pts[leaf],"o",color=ink,ms=1.9,zorder=3)
            ax.text(pts[leaf][0],.60,str(leaf),ha="center",va="center",fontsize=PT_SMALL,color=ink)
        ax.text(x0+w/2,.50,label,ha="center",va="center",fontsize=PT_ANNOT,color=ink)
    # bottom strip: the route budgets on the first tree; filled nodes are the K channels
    for i,k in enumerate(BUDGETS):
        item=by_id[f"balanced_contiguous_k{k}"]
        children,root,order=_tree(item)
        x0=i*(slot_w+gap)+.012;w=slot_w-.024
        pts=_layout(children,root,order,x0,w,.15,.40)
        for n,kids in children.items():
            for c in kids:
                ax.plot([pts[n][0],pts[c][0]],[pts[n][1],pts[c][1]],color=edge,lw=LW_HAIR,zorder=2)
        cut=_cut_nodes(item,children,root)
        assert len(cut)==k
        for n in cut:
            ax.plot(*pts[n],"o",color=ink,ms=3.2,zorder=3)
        ax.text(x0+w/2,.03,f"K = {k}",ha="center",va="bottom",fontsize=PT_ANNOT,color=ink)


def protocol_panel(ax):
    boxes=[(.02,.69,.96,.25,"Development tasks\nscore-to-loss scale"),
           (.02,.36,.96,.25,"Calibration samples\nsealed scores and choices"),
           (.02,.03,.96,.25,"Held-out tasks\ntraining and regret")]
    for x,y,w,h,label in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor=COLORS["panel_bg"],edgecolor=COLORS["edge"],lw=LW_HAIR))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=PT_ANNOT,linespacing=1.3)
    for y in [.655,.325]:
        ax.annotate("",xy=(.50,y-.045),xytext=(.50,y+.035),
                    arrowprops={"arrowstyle":"-|>","color":COLORS["mute"],"lw":LW_EDGE})


def regret_panel(ax,summary,seeds):
    """Four policy rows, both arms per row, twenty whole-seed means per cloud."""
    rng=np.random.default_rng(JITTER_SEED)
    dodge=.19;band=.13
    for y,policy in enumerate(POLICIES):
        for arm,color,sign in zip(ARMS,ARM_COLORS,[-1,1]):
            row=summary[summary.arm.eq(arm)&summary.policy.eq(policy)].iloc[0]
            values=seeds[seeds.arm.eq(arm)&seeds.policy.eq(policy)].sort_values("seed").regret.to_numpy()
            assert len(values)==20,(arm,policy,len(values))
            yc=y+sign*dodge
            ax.scatter(values,yc+rng.uniform(-band,band,len(values)),s=6,color=color,
                       alpha=.45,linewidths=0,zorder=3)
            mean=float(row.mean_regret)
            ax.errorbar(mean,yc,xerr=[[mean-row.ci95_low],[row.ci95_high-mean]],
                        fmt="D",ms=3.6,mfc="white",mec=color,color=color,lw=LW_DATA,
                        capsize=2,zorder=5,label=ARM_LABELS[arm] if y==0 else None)
    ax.axvline(0,color=COLORS["mute"],ls="--",lw=LW_REF,zorder=1)
    ax.set_yticks(range(len(POLICIES)),[LABELS[p] for p in POLICIES])
    ax.tick_params(axis="y",length=0)
    ax.set_ylim(len(POLICIES)-.5,-.5)
    ax.set_xlim(-.004,.15);ax.set_xticks(np.arange(0,.151,.025))
    ax.set_xlabel("Test loss + cost regret (excess over best trained candidate)",fontsize=PT_LABEL)
    ax.legend(loc="upper right",frameon=False,fontsize=PT_SMALL,handletextpad=.4,
              title=None,borderaxespad=.2)


def budget_panel(ax,policies):
    selected=policies[policies.policy.eq("moment_selector")]
    ranks=sorted(selected["rank"].unique())
    assert ranks==BUDGETS
    # retrospectively best budget: verified identical to the imposed rank in
    # every task-arm case, so one reference line stands for both arms.
    assert (selected.oracle_budget_k==selected["rank"]).all()
    xs=np.arange(len(ranks))
    ax.plot(xs,ranks,color=COLORS["point_mlp"],lw=LW_REF,ls="--",zorder=2)
    for arm,color,offset in zip(ARMS,ARM_COLORS,[-.17,.17]):
        frame=selected[selected.arm.eq(arm)]
        per_seed=frame.groupby(["rank","seed"]).budget_k.mean()
        means=frame.groupby("rank").budget_k.mean().loc[ranks].to_numpy()
        for x,rank in zip(xs,ranks):
            vals=per_seed.loc[rank].to_numpy()
            assert len(vals)==20
            # seed-block means fall on quarter steps, so ties are spread
            # side by side (a dot histogram) instead of piling up
            px=[];py=[]
            for value,count in zip(*np.unique(np.round(vals,6),return_counts=True)):
                half=min(.085,.013*(count-1))
                px.extend(x+offset+np.linspace(-half,half,count));py.extend([value]*count)
            ax.scatter(px,py,s=6,color=color,alpha=.45,linewidths=0,zorder=3)
        ax.plot(xs+offset,means,marker="o",ms=4.2,mfc="white",mec=color,mew=LW_DATA,
                color=color,lw=LW_DATA,zorder=5)
    ax.set_xticks(xs,[str(r) for r in ranks]);ax.set_yticks(range(1,9))
    ax.set_ylim(.3,8.5);ax.set_xlim(-.5,len(ranks)-.5)
    ax.set_xlabel("Task rank");ax.set_ylabel("Route budget K of selected candidate")


def horizon_panel(ax):
    first=pd.read_csv(SOURCE/"first_step_correlation_summary.csv")
    first_seed=pd.read_csv(SOURCE/"first_step_within_task_correlations.csv")
    final=pd.read_csv(SOURCE/"within_task_correlations.csv")
    jitter=np.random.default_rng(JITTER_SEED)
    for arm,color,offset in zip(ARMS,ARM_COLORS,[-.12,.12]):
        row=first[first.arm.eq(arm)&first.endpoint.eq("actual_training_first_step_test_decrease")].iloc[0]
        first_vals=(first_seed[first_seed.arm.eq(arm)
                    &first_seed.endpoint.eq("actual_training_first_step_test_decrease")]
                    .groupby("seed").spearman_rho.mean().to_numpy())
        vals=final[final.arm.eq(arm)].groupby("seed").utility_vs_negative_loss_rho.mean().to_numpy()
        assert len(first_vals)==20 and len(vals)==20
        assert np.isclose(first_vals.mean(),row.mean_within_task_rho,atol=1e-6)
        rng=np.random.default_rng(20260905)
        lo,hi=np.quantile(rng.choice(vals,(10000,len(vals)),replace=True).mean(axis=1),[.025,.975])
        for x,v in zip([0+offset,1+offset],[first_vals,vals]):
            ax.scatter(x+jitter.uniform(-.045,.045,len(v)),v,s=6,color=color,alpha=.4,
                       linewidths=0,zorder=3)
        ax.errorbar([0+offset,1+offset],[row.mean_within_task_rho,vals.mean()],
                    yerr=[[row.mean_within_task_rho-row.ci95_low,vals.mean()-lo],
                          [row.ci95_high-row.mean_within_task_rho,hi-vals.mean()]],
                    color=color,marker="o",ms=4.2,mfc="white",mec=color,mew=LW_DATA,
                    lw=LW_DATA,capsize=2,zorder=5)
    ax.set_xticks([0,1],["first-step test-loss\ndecrease","final test-loss\nranking"])
    ax.set_ylabel("Mean within-task Spearman ρ")
    ax.set_yticks(np.arange(.5,1.01,.1))
    ax.set_ylim(.5,1.01);ax.set_xlim(-.4,1.4)


def build():
    manifest=json.loads((SOURCE/"candidate_manifest.json").read_text())
    summary=pd.read_csv(SOURCE/"policy_summary.csv")
    seeds=pd.read_csv(SOURCE/"policy_seed_means.csv")
    policies=pd.read_csv(SOURCE/"policy_outcomes.csv")
    assert set(POLICIES).issubset(summary.policy.unique())
    canvas=NativeCanvas(490/72,3,row_weights=[118,118,128],
                        hgutter_pt=34,vgutter_pt=32,
                        margins=Margins(left=43,right=12,top=20,bottom=35))
    a=canvas.panel("A",0,0,7,schematic=True)
    b=canvas.panel("B",0,7,5,schematic=True)
    c=canvas.panel("C",1,0,12,grid="x")
    d=canvas.panel("D",2,0,7,grid="y")
    e=canvas.panel("E",2,7,5,grid="y")
    candidate_panel(a,manifest);protocol_panel(b)
    regret_panel(c,summary,seeds)
    budget_panel(d,policies);horizon_panel(e)
    return canvas.save(OUT,name="figure_S35_panels_A-F")


if __name__=="__main__":
    build()
