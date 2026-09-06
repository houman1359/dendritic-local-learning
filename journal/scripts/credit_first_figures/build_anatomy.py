#!/usr/bin/env python3
"""Selective main anatomy sheet: measured arbor, common mode, residuals, cost."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
JOURNAL=HERE.parents[1]
sys.path.insert(0,str(JOURNAL / "scripts"))
import build_main_figure_07 as anatomy
from figure_canvas import (COLORS,LW_EDGE,LW_ERR,LW_HAIR,LW_REF,MARKER_MS,
                           PT_ANNOT,PT_LABEL,PT_LEGEND,PT_SMALL,Margins,NativeCanvas,style_panel)
from journal_style import style_direct_color_labels

SOURCE=JOURNAL / "source_data"
COMMON=SOURCE / "anatomy_commonmode"
RECORDS=SOURCE / "credit_first_figures"
OUT=JOURNAL / "figures/components/credit_first_figure_06.pdf"
METHODS=["common + ancestry","common + random routes","common + depth bins",
         "common + shuffled routes","common + surrogate ancestry","common-constrained SVD"]
LABELS=["Ancestry","Random routes","Depth bins","Shuffled routes","Surrogate tree","SVD oracle"]
COHORTS=["original8","v661","pinky"]


def energy_partition(ax, table):
    focus=table[table.channels.eq(8)].groupby("method").mean(numeric_only=True).loc[METHODS]
    total=focus.total_capture.to_numpy()
    spatial=focus.incremental_total_capture.to_numpy()
    common=total-spatial
    y=np.arange(len(METHODS))[::-1]
    ax.barh(y,common,color=COLORS["scalar"],height=.61,edgecolor="white",lw=LW_HAIR)
    ax.barh(y,spatial,left=common,color=COLORS["shunting"],height=.61,edgecolor="white",lw=LW_HAIR)
    ax.barh(y,1-total,left=total,color=COLORS["grid"],height=.61,edgecolor="white",lw=LW_HAIR)
    ax.set_yticks(y,LABELS)
    ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel("Fraction of total response energy")
    ax.set_ylim(-.55,6.1)
    handles=[Patch(facecolor=COLORS[k],label=l) for k,l in [("scalar","Common"),("shunting","Spatial"),("grid","Uncaptured")]]
    ax.legend(handles=handles,loc="upper center",bbox_to_anchor=(.45,.995),ncol=3,
              frameon=False,handlelength=.8,handletextpad=.35,columnspacing=.7,fontsize=PT_SMALL)
    style_panel(ax)
    ax.tick_params(axis="y",length=0,labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)
    return focus


def contrast_forest(ax, report):
    rows=[]
    for y,method,label in zip([3,2,1,0],METHODS[1:5],LABELS[1:5]):
        item=next(v for v in report["comparisons"] if v["metric"]=="residual_capture" and v["control"]==method)
        mean=100*item["mean_difference"];lo,hi=100*np.array(item["ci95"])
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt="o",color=COLORS["shunting"],
                    ms=MARKER_MS,elinewidth=LW_ERR,capsize=2)
        rows.append(dict(control=method,mean_pp=mean,low_pp=lo,high_pp=hi,n_cells=item["n_cells"],
                         positive_cells=item["cells_positive"]))
    ax.axvline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    ax.set_yticks([3,2,1,0],LABELS[1:5])
    ax.set_ylim(-.6,3.6);ax.set_xlim(-1.5,34.5);ax.set_xticks([0,10,20,30])
    ax.set_xlabel("Ancestry advantage in spatial-residual capture (percentage points)")
    style_panel(ax,grid="x")
    ax.tick_params(axis="y",length=0,labelsize=PT_ANNOT)
    ax.spines["left"].set_visible(False)
    return pd.DataFrame(rows)


def cohort_points(ax,tables):
    rng=np.random.default_rng(202609061)
    rows=[]
    for y,cohort,label in zip([2,1,0],COHORTS,["Development\n8 cells","Disjoint cells\n47 cells","Second mouse\n8 cells"]):
        table=tables[cohort]
        values=table[table.channels.eq(8)&table.method.eq(METHODS[0])]
        for metric,offset,filled in [("total_capture",.14,True),("residual_capture",-.14,False)]:
            data=values[metric].to_numpy()
            draws=rng.choice(data,size=(20000,len(data)),replace=True).mean(axis=1)
            mean=data.mean();low,high=np.quantile(draws,[.025,.975])
            ax.errorbar(mean,y+offset,xerr=[[mean-low],[high-mean]],fmt="o",color=COLORS["shunting"],
                        markerfacecolor=COLORS["shunting"] if filled else "white",markeredgewidth=LW_ERR,
                        ms=MARKER_MS,elinewidth=LW_ERR,capsize=2)
            rows.append(dict(cohort=cohort,metric=metric,mean=mean,low=low,high=high,n_cells=len(data)))
    ax.set_yticks([2,1,0],["Development\n8 cells","Disjoint cells\n47 cells","Second mouse\n8 cells"])
    ax.set_ylim(-.5,3.25);ax.set_xlim(.1,1.045);ax.set_xticks([.25,.5,.75,1])
    ax.set_xlabel("Ancestry + common capture")
    handles=[plt.Line2D([],[],marker="o",ls="none",color=COLORS["shunting"],mfc=c,label=l)
             for c,l in [(COLORS["shunting"],"Total"),("white","Spatial residual")]]
    ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(-.025,.995),ncol=1,frameon=False,
              columnspacing=.6,handletextpad=.3,fontsize=PT_SMALL)
    style_panel(ax,grid="x");ax.tick_params(axis="y",length=0,labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)
    return pd.DataFrame(rows)


def costs(ax,focus):
    ax.set_axis_off()
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.025,.94,"Dictionary",fontsize=PT_LABEL,ha="left")
    ax.text(.62,.94,"Rank",fontsize=PT_LABEL,ha="right")
    ax.text(.97,.94,"Wiring",fontsize=PT_LABEL,ha="right")
    ax.plot([.025,.97],[.885,.885],color=COLORS["edge"],lw=LW_HAIR)
    for y,method,label in zip(np.linspace(.80,.22,6),METHODS,LABELS):
        row=focus.loc[method]
        color=COLORS["shunting"] if method==METHODS[0] else COLORS["ink"]
        ax.text(.025,y,label,fontsize=PT_LABEL,ha="left",va="center",color=color)
        ax.text(.62,y,f"{row.dictionary_rank:.2f}",fontsize=PT_LABEL,ha="right",va="center",color=color)
        ax.text(.97,y,f"{100*row.wiring_density:.1f}%",fontsize=PT_LABEL,ha="right",va="center",color=color)
    ax.text(.50,.025,"47-cell means; K = 8 includes the broadcast",ha="center",fontsize=PT_SMALL,color=COLORS["mute"])


def main():
    RECORDS.mkdir(exist_ok=True)
    tables={c:pd.read_csv(COMMON / c / "cell_method_summary.csv") for c in COHORTS}
    reports={c:json.loads((COMMON / c / "summary.json").read_text()) for c in COHORTS}
    canvas=NativeCanvas(526/72,3,row_weights=[150,126,132],hgutter_pt=42,vgutter_pt=44,
                        margins=Margins(left=38,right=14,top=24,bottom=35))
    a=canvas.panel("A",0,0,6,title="Reconstructed arbor and mapped contacts",schematic=True,lock=False)
    b=canvas.panel("B",0,6,6,title="Beyond a common broadcast: 47 cells")
    c=canvas.panel("C",1,0,12,title="Ancestry explains the remaining spatial field")
    d=canvas.panel("D",2,0,6,title="Capacity across cohorts")
    e=canvas.panel("E",2,6,6,title="Realized rank and wiring cost",schematic=True,lock=False)
    anatomy.panel_arbor(a)
    focus=energy_partition(b,tables["v661"])
    effects=contrast_forest(c,reports["v661"])
    # The full-width forest's left reserve is explicit because its long y
    # categories are independent of the schematic/point column above/below.
    canvas.declare_reserve("C",left=65)
    cohort=cohort_points(d,tables)
    costs(e,focus)
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    right_letters=[item["art"] for item in canvas._letters if item["letter"] in "BE"]
    shared_x=min(art.get_position()[0] for art in right_letters)
    for art in right_letters:art.set_position((shared_x,art.get_position()[1]))
    problems=canvas.save(OUT,name="credit_first_figure_06",dpi=180,lock=False)
    effects.to_csv(RECORDS / "figure_06_residual_contrasts.csv",index=False)
    cohort.to_csv(RECORDS / "figure_06_cohort_points.csv",index=False)
    focus.reset_index().to_csv(RECORDS / "figure_06_primary_means.csv",index=False)
    cell,*_=anatomy.morphology_geometry()
    files=[Path(__file__),Path(anatomy.__file__),JOURNAL / "scripts/figure_canvas.py",JOURNAL / "scripts/journal_style.py",
           SOURCE / "figure3/segment_metrics.csv",COMMON / "protocol_freeze.json"]
    files += [COMMON / c / "cell_method_summary.csv" for c in COHORTS]+[COMMON / c / "summary.json" for c in COHORTS]
    mapping={"A":f"Public build_main_figure_07.panel_arbor; median-sized original-cohort example, root {int(cell.root_id.iloc[0])}; actual PCA-projected segment coordinates and mapped E/I contact area",
             "B":"Frozen common-mode comparison, v661 47-cell means at K=8; total energy partitioned into common broadcast, additional spatial capture, and uncaptured response",
             "C":"Frozen v661 paired residual-capture contrasts and original 95% cell-bootstrap intervals; all four spatial controls",
             "D":"Common+ancestry total and residual capture at K=8 per cohort; 20,000-draw descriptive cell-bootstrap intervals with seed 202609061",
             "E":"Frozen v661 actual ranks and nonzero densities at K=8; 200 randomized controls averaged within each cell before cohort summaries"}
    payload=dict(panel_sources=mapping,source_sha256={str(p.relative_to(JOURNAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                 layout_findings=problems,scope="Modeled response capacity on measured anatomy; no evidence of endogenous biological route usage. Original8/v661 are the same animal; Pinky is one second animal.")
    (RECORDS / "figure_06_sources.json").write_text(json.dumps(payload,indent=2)+"\n")
    caption=("**Ancestry dictionaries capture spatial response structure beyond a shared broadcast.** "
       "**A**, An actual reconstructed arbor from the original eight-cell cohort, selected by median segment count. "
       "Segment color encodes the balance of mapped excitatory (E) and inhibitory (I) contact area; width encodes "
       "total mapped area in five logarithmic levels. The skeleton is shown in an isotropically scaled principal-plane "
       "projection; the bar denotes 50 µm. **B**, Energy captured in exact reciprocal passive-cable responses to "
       "modeled focal shunts, evaluated in 47 disjoint v661 cells from the same MICrONS mouse. Every dictionary "
       "contains a constant broadcast plus seven spatial profiles. Bars separate the common component, additional "
       "spatial capture and uncaptured energy; values are cell means. **C**, Ancestry advantages in the spatial "
       "residual after removing that weighted broadcast. Points and bars show paired cell means and retained 95% "
       "cell-bootstrap intervals. Random routes sample anatomical sites; shuffled routes preserve the selected "
       "columns' nonzero counts; surrogate trees preserve segment depths and parent out-degrees. **D**, Ancestry "
       "capture in the original eight cells (post hoc development), disjoint v661 cells, and eight eligible Pinky "
       "cells from a second mouse. Filled and open symbols denote total and residual energy fractions, respectively; "
       "bars are descriptive 95% cell-bootstrap intervals. Pinky's near-saturation concerns 9–13 excitatory-bearing "
       "sites per cell. **E**, Actual dictionary rank and nonzero coefficients as a percentage of dense eight-column "
       "wiring in v661. All randomized controls average 200 draws per cell. Equal channel count does not imply "
       "equal rank or wiring; ancestry and shuffled routes have identical nonzero counts. The common-constrained "
       "SVD oracle is a representational ceiling. Capture uses squared response energy weighted by excitatory "
       "synapse size; residual capture uses the energy remaining after the common projection. Cell intervals "
       "describe within-cohort consistency, and the modeled perturbations do not establish biological use of "
       "these learning-signal routes.\n")
    (RECORDS / "figure_06_caption.md").write_text(caption)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
