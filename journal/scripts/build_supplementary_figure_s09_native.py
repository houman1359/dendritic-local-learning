#!/usr/bin/env python3
"""Retained checkpoint geometry with explicit scope and within-rule ranks."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from figure_canvas import (COLORS,LW_DATA,LW_REF,PT_SMALL,Margins,NativeCanvas)
from analyze_prospective_learning_results import bootstrap_ci

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"source_data/prospective_input_validity"
OUT=ROOT/"figures/supplementary/figure_S09_panels_A-D.pdf"
FAMILIES=["global_scalar_available","ancestry_available","exact_transport"]
LABELS=["MW scalar","Neuron","Exact path"]
TONES=[COLORS["scalar"],COLORS["per_soma"],COLORS["oracle"]]


def build():
    data=pd.read_csv(SOURCE/"mechanism_checkpoint_rows_valid.csv")
    data=data[np.isclose(data.relative_step,1e-5)&data.feedback_family.isin(FAMILIES)]
    cv=NativeCanvas(355/72,2,hgutter_pt=32,vgutter_pt=52,
                    margins=Margins(left=43,right=12,top=36,bottom=32))
    cv.fig.text(.5,.985,"120 trained checkpoints; matched update norms; relative step 10⁻⁵",
                ha="center",va="top",fontsize=PT_SMALL,color=COLORS["mute"])
    axes=[]
    for letter,row,col,title in [("A",0,0,"Gradient direction"),("B",0,6,"Gradient capture"),
                                  ("C",1,0,"One-step progress"),("D",1,6,"Cross-rule geometric association")]:
        axes.append(cv.panel(letter,row,col,6,title=title,grid="y"))
    metrics=["gradient_cosine","gradient_scaled_capture","norm_matched_fraction_of_exact"]
    ylabs=["gradient cosine","scaled gradient capture","fraction of exact progress"]
    for ax,metric,ylabel in zip(axes[:3],metrics,ylabs):
        arrays=[data[data.feedback_family.eq(f)][metric].to_numpy() for f in FAMILIES]
        assert all(len(v)==120 for v in arrays)
        boxes=ax.boxplot(arrays,positions=range(3),widths=.48,patch_artist=True,
                          showfliers=False,medianprops={"color":"white","linewidth":1.1})
        for j,(vals,box,color) in enumerate(zip(arrays,boxes["boxes"],TONES)):
            box.set_facecolor(color);box.set_alpha(.6)
            old_title={"gradient_cosine":"Gradient direction", "gradient_scaled_capture":"Gradient capture",
                       "norm_matched_fraction_of_exact":"One-step learning"}[metric]
            mean,lo,hi=bootstrap_ci(vals,seed=700+len(old_title))
            ax.errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt="D",mfc="white",color=COLORS["ink"],ms=3,lw=LW_DATA,capsize=2)
        ax.set_xticks(range(3),LABELS);ax.set_ylabel(ylabel)
        ax.axhline(0,color=COLORS["mute"],ls="--",lw=LW_REF)
    axes[0].set_ylim(-.16,1.1);axes[1].set_ylim(-.03,1.1);axes[2].set_ylim(-1.6,1.18)
    for j,family in enumerate(FAMILIES):
        vals=data[data.feedback_family.eq(family)]
        axes[2].text(j,-1.47,f"{int(vals.norm_matched_is_descent.sum())}/120 descent",
                      ha="center",fontsize=PT_SMALL,color=COLORS["mute"])
    d=axes[3]
    for family,label,color in zip(FAMILIES[:2],LABELS[:2],TONES[:2]):
        part=data[data.feedback_family.eq(family)]
        rho=spearmanr(part.gradient_cosine,part.norm_matched_fraction_of_exact).statistic
        d.scatter(part.gradient_cosine,part.norm_matched_fraction_of_exact,s=8,color=color,alpha=.4,
                    label=f"{label}: ρ={rho:.2f}")
    d.axhline(0,color=COLORS["mute"],ls="--",lw=LW_REF)
    d.set_xlabel("cosine with exact gradient");d.set_ylabel("fraction of exact progress")
    d.set_ylim(-1.6,1.18);d.legend(loc="lower right",frameon=False,fontsize=PT_SMALL)
    d.text(.03,.97,"legend: within-rule association",va="top",transform=d.transAxes,
            fontsize=PT_SMALL,color=COLORS["mute"])
    return cv.save(OUT,name="figure_S09_panels_A-D")

if __name__=="__main__":build()
