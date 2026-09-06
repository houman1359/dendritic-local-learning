#!/usr/bin/env python3
"""Original prospective morphology selection, retained as Supplementary Figure S35.

The assembly map assigns the final publication number (Figure 6). Every
outcome comes from the sealed prospective experiment; no training or model
selection occurs in this renderer. Strong baselines and negative results
receive the same visual prominence as the selector.
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
POLICIES=["moment_selector","rank_only","development_best","uniform_random_expectation"]
LABELS={"moment_selector":"moment selector","rank_only":"rank-only",
        "development_best":"fixed / max-budget","uniform_random_expectation":"random expectation"}


def candidate_panel(ax,manifest,protocol):
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    for x0,name in [(.00,"balanced_contiguous"),(.52,"comb_contiguous")]:
        item=next(c for c in manifest if c["tree_id"]==name and c["budget_k"]==4)
        positions={int(k):np.asarray(v,float) for k,v in item["positions"].items()}
        xy=np.vstack(list(positions.values()));low=xy.min(0);span=np.ptp(xy,axis=0)
        points={k:np.array([x0+.04+.39*(p[0]-low[0])/span[0],
                           .44+.51*(p[1]-low[1])/span[1]]) for k,p in positions.items()}
        for a,b,length in item["edges"]:
            u,v=points[int(a)],points[int(b)]
            ax.plot([u[0],v[0]],[u[1],v[1]],color=COLORS["dend"],lw=LW_EDGE)
        for leaf in range(8):
            p=points[leaf]
            ax.plot(*p,"o",color=COLORS["additive"],ms=2.5)
        ax.text(x0+.235,.34,"balanced" if x0==0 else "comb",ha="center",fontsize=PT_LABEL)
    ax.text(.00,.18,"4 balanced leaf assignments + 1 comb",fontsize=PT_ANNOT)
    ax.text(.00,.065,"K = 1, 2, 4, 8; 20 costed candidates",fontsize=PT_ANNOT)


def protocol_panel(ax,protocol):
    boxes=[(.015,.69,.96,.24,"Development tasks\nfit score-to-loss scale"),
           (.015,.36,.96,.24,"Independent calibration samples\nseal all candidate scores and choices"),
           (.015,.03,.96,.24,"Held-out tasks / 20 new seeds\ntrain every candidate, then assess regret")]
    for x,y,w,h,label in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor=COLORS["panel_bg"],edgecolor=COLORS["edge"],lw=LW_HAIR))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=PT_ANNOT,linespacing=1.3)
    for y in [.655,.325]:
        ax.annotate("",xy=(.495,y-.045),xytext=(.495,y+.035),
                    arrowprops={"arrowstyle":"-|>","color":COLORS["mute"],"lw":LW_EDGE})


def regret_panel(ax,arm,summary,seeds):
    color=ARM_COLORS[ARMS.index(arm)]
    for y,policy in enumerate(POLICIES):
        row=summary[summary.arm.eq(arm)&summary.policy.eq(policy)].iloc[0]
        values=seeds[seeds.arm.eq(arm)&seeds.policy.eq(policy)].sort_values("seed").regret.to_numpy()
        tone=color if policy=="moment_selector" else COLORS["point_mlp"]
        ax.scatter(values,y+np.linspace(-.12,.12,len(values)),s=9,color=tone,alpha=.4,zorder=3)
        mean=float(row.mean_regret)
        ax.errorbar(mean,y,xerr=[[mean-row.ci95_low],[row.ci95_high-mean]],
                    fmt="D",ms=3.7,color=tone,lw=LW_DATA,capsize=2,zorder=4)
        ax.text(.003,y-.30,LABELS[policy],fontsize=PT_ANNOT,color=tone)
    ax.axvline(0,color=COLORS["mute"],ls="--",lw=LW_REF)
    ax.set_yticks([]);ax.set_ylim(3.35,-.65);ax.set_xlim(-.005,.20)
    ax.set_xlabel("test loss + cost regret",fontsize=PT_LABEL)



def budget_panel(ax,policies):
    selected=policies[policies.policy.eq("moment_selector")]
    for arm,color in zip(ARMS,ARM_COLORS):
        frame=selected[selected.arm.eq(arm)]
        group=frame.groupby("rank")[["budget_k","oracle_budget_k"]].mean()
        xs=np.arange(len(group))
        ax.plot(xs,group.budget_k,marker="o",color=color,lw=LW_DATA,
                label="feedback only" if arm==ARMS[0] else "joint transfer")
        ax.plot(xs,group.oracle_budget_k,marker="D",mfc="white",color=color,
                lw=LW_REF,ls="--")
    ax.set_xticks(range(4),["1","2","4","8"]);ax.set_yticks([1,2,4,8])
    ax.set_ylim(.5,8.6);ax.set_xlabel("task rank");ax.set_ylabel("mean route budget K")
    ax.text(.02,.98,"solid: selector\ndashed: retrospective best",
            transform=ax.transAxes,va="top",fontsize=PT_SMALL,color=COLORS["mute"])
    ax.legend(loc="lower right",frameon=False,fontsize=PT_SMALL)


def rotation_panel(ax,outcomes,protocol):
    noise=max(protocol["noise_sds"])
    frame=outcomes[outcomes["rank"].eq(4)&outcomes.budget_k.eq(4)&np.isclose(outcomes.noise_sd,noise)]
    tree_order=[c["name"] for c in protocol["candidate_trees"]]
    columns=[(arm,angle) for arm in ARMS for angle in protocol["angles_pi"]["confirmatory"]]
    means=frame.groupby(["tree_id","arm","angle_pi"]).penalized_test_loss.mean()
    matrix=np.array([[means.loc[(tree,arm,angle)] for arm,angle in columns] for tree in tree_order])
    # Column centering displays a native loss difference, preserving units
    # while making between-tree sensitivity legible in the two linear arms.
    excess=matrix-matrix.min(axis=0,keepdims=True)
    image=ax.imshow(excess,aspect="auto",cmap="Blues",vmin=0,vmax=max(.001,excess.max()))
    for row in range(5):
        for col in range(4):
            ax.text(col,row,f"{excess[row,col]:.2f}",ha="center",va="center",fontsize=PT_SMALL,
                    color="white" if excess[row,col]>.58*excess.max() else COLORS["ink"])
    ax.set_yticks(range(5),["balanced 1","balanced 2","balanced 3","balanced 4","comb"])
    ax.set_xticks(range(4),["π/8","3π/8","π/8","3π/8"])
    ax.tick_params(length=0)
    ax.axvline(1.5,color="white",lw=2.5)
    ax.text(.5,-.22,"feedback",ha="center",transform=ax.get_xaxis_transform(),fontsize=PT_SMALL)
    ax.text(2.5,-.22,"joint",ha="center",transform=ax.get_xaxis_transform(),fontsize=PT_SMALL)
    ax.text(.5,1.035,f"r = K = 4; noise SD = {noise:g}",ha="center",transform=ax.transAxes,
            fontsize=PT_SMALL,color=COLORS["mute"])
    ax.set_xlabel("rotation angle (fixed spectrum)",labelpad=23)


def horizon_panel(ax):
    first=pd.read_csv(SOURCE/"first_step_correlation_summary.csv")
    final=pd.read_csv(SOURCE/"within_task_correlations.csv")
    for arm,color,offset in zip(ARMS,ARM_COLORS,[-.10,.10]):
        row=first[first.arm.eq(arm)&first.endpoint.eq("actual_training_first_step_test_decrease")].iloc[0]
        vals=final[final.arm.eq(arm)].groupby("seed").utility_vs_negative_loss_rho.mean().to_numpy()
        rng=np.random.default_rng(20260905)
        lo,hi=np.quantile(rng.choice(vals,(10000,len(vals)),replace=True).mean(axis=1),[.025,.975])
        ax.errorbar([0+offset,1+offset],[row.mean_within_task_rho,vals.mean()],
                    yerr=[[row.mean_within_task_rho-row.ci95_low,vals.mean()-lo],
                          [row.ci95_high-row.mean_within_task_rho,hi-vals.mean()]],
                    color=color,marker="o",lw=LW_DATA,capsize=2)
    ax.set_xticks([0,1],["first test-loss\ndecrease","final test-loss\nranking"])
    ax.set_ylabel("mean within-task Spearman ρ")
    ax.set_ylim(0,1.04);ax.set_xlim(-.35,1.35)
    ax.text(.98,.08,"20 seed blocks; 95% CI",ha="right",transform=ax.transAxes,
            fontsize=PT_SMALL,color=COLORS["mute"])


def build():
    protocol=json.loads((SOURCE/"protocol.json").read_text())
    manifest=json.loads((SOURCE/"candidate_manifest.json").read_text())
    summary=pd.read_csv(SOURCE/"policy_summary.csv")
    seeds=pd.read_csv(SOURCE/"policy_seed_means.csv")
    policies=pd.read_csv(SOURCE/"policy_outcomes.csv")
    outcomes=pd.read_csv(SOURCE/"candidate_outcomes.csv")
    assert set(POLICIES).issubset(summary.policy.unique())
    canvas=NativeCanvas(490/72,3,row_weights=[125,135,135],
                        hgutter_pt=34,vgutter_pt=44,
                        margins=Margins(left=43,right=12,top=20,bottom=35))
    a=canvas.panel("A",0,0,6,schematic=True,title="Explicit morphology / routing candidates")
    b=canvas.panel("B",0,6,6,schematic=True,title="Selection is sealed before training")
    c=canvas.panel("C",1,0,6,title="Feedback only: strong baselines win",grid="x")
    d=canvas.panel("D",1,6,6,title="Joint transfer: strong baselines win",grid="x")
    e=canvas.panel("E",2,0,6,title="Selected budgets differ from the best",grid="y")
    f=canvas.panel("F",2,6,6,title="Local agreement is stronger",grid="y")
    candidate_panel(a,manifest,protocol);protocol_panel(b,protocol)
    regret_panel(c,ARMS[0],summary,seeds);regret_panel(d,ARMS[1],summary,seeds)
    budget_panel(e,policies);horizon_panel(f)
    return canvas.save(OUT,name="figure_S35_panels_A-F")


if __name__=="__main__":
    build()
