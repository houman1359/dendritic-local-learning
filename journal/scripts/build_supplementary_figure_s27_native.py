#!/usr/bin/env python3
"""Second-animal routing results with explicit candidate-size diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_REF, PT_ANNOT,
                           PT_SMALL, Margins, NativeCanvas)
from analyze_pinky_v185_replication import METHOD_ORDER, METHOD_STYLE, bootstrap_mean

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"source_data/pinky_v185_replication"
OUT=ROOT/"figures/supplementary/figure_S27_panels_A-C.pdf"


def build():
    cohort=pd.read_csv(SOURCE/"cohort_manifest.csv")
    cells=pd.read_csv(SOURCE/"routing/cell_metrics.csv")
    cells=cells[cells.qc_included]
    curves=pd.read_csv(SOURCE/"routing/feedback_compression_curves.csv")
    cv=NativeCanvas(395/72,2,hgutter_pt=32,vgutter_pt=62,
                    margins=Margins(left=46,right=14,top=20,bottom=38))
    a=cv.panel("A",0,0,6,title="Second mouse: x–y projection")
    b=cv.panel("B",0,6,6,title="Modeled residual; mean ± SEM",grid="y")
    c=cv.panel("C",1,0,6,title="Residual reduction per coefficient",grid="y")
    d=cv.panel("D",1,6,6,title="Candidate routes and field rank",grid="y")
    colors=np.where(cohort.root_id.isin(cells.root_id),COLORS["dend"],COLORS["mute"])
    a.scatter(cohort.x_nm/1000,cohort.y_nm/1000,c=colors,s=20)
    for row in cohort.itertuples():
        a.text(row.x_nm/1000+1.2,row.y_nm/1000,str(row.selection_order+1),fontsize=PT_SMALL)
    a.set_xlabel("volume x (µm)");a.set_ylabel("volume y (µm)");a.set_ylim(150,315)
    a.text(.02,.96,f"{len(cells)} qualifying / {len(cohort)} selected trees",transform=a.transAxes,
            va="top",fontsize=PT_SMALL,color=COLORS["mute"])
    for method in METHOD_ORDER:
        frame=curves[curves.method.eq(method)].groupby("channels").residual.agg(["mean","sem"])
        color,ls,marker=METHOD_STYLE[method]
        b.errorbar(frame.index,frame["mean"],yerr=frame["sem"].fillna(0),color=color,
                    ls=ls,marker=marker,ms=3.5,lw=LW_DATA,capsize=2,
                    label=method.replace("morphology-aware paths","ancestry routes"))
    counts=curves[curves.method.eq("dense PCA oracle")].groupby("channels").root_id.nunique()
    b.set_xscale("log",base=2);b.set_xticks(counts.index,[f"{k}\n(n={n})" for k,n in counts.items()])
    b.set_ylim(-.03,1.1);b.set_xlabel("route budget K; qualifying cells")
    b.set_ylabel("weighted relative residual norm")
    handles, labels = b.get_legend_handles_labels()
    cv.fig.legend(handles, labels, loc="center", bbox_to_anchor=(.55,.50), ncol=3, frameon=False, fontsize=PT_SMALL)
    fixed=curves[curves.channels.eq(4)].copy()
    fixed["value"]=(1-fixed.residual)/fixed.wiring_nonzeros*1000
    for i,method in enumerate(["dense PCA oracle","morphology-aware paths","random paths"]):
        vals=fixed[fixed.method.eq(method)].value.to_numpy();color,ls,marker=METHOD_STYLE[method]
        mean,lo,hi=bootstrap_mean(vals,20260840+i)
        c.scatter(i+np.linspace(-.07,.07,len(vals)),vals,s=9,color=color,alpha=.4)
        c.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt=marker,color=color,lw=LW_DATA,capsize=2)
    c.set_xticks(range(3),["dense PCA","ancestry","random"])
    c.set_ylabel("residual reduction /\n1,000 nonzeros");c.set_ylim(top=315)
    c.text(.98,.96,"K=4; 10 cells; 95% CI",ha="right",va="top",transform=c.transAxes,
            fontsize=PT_SMALL,color=COLORS["mute"])
    d.scatter(cells.n_i_segments,cells.credit_kernel_participation_rank,s=22,color=COLORS["dend"])
    for row in cells.itertuples():
        d.annotate(str(row.selection_order+1),(row.n_i_segments,row.credit_kernel_participation_rank),
                    xytext=(3,2),textcoords="offset points",fontsize=PT_SMALL)
    d.set_xlabel("candidate routes (I-bearing segments)")
    d.set_ylabel("kernel participation rank");d.set_ylim(1.2,2.85)
    d.text(.03,.97,f"{int(cells.n_segments.min())}–{int(cells.n_segments.max())} reconstructed segments / tree",
            transform=d.transAxes,va="top",fontsize=PT_SMALL,color=COLORS["mute"])
    return cv.save(OUT,name="figure_S27_panels_A-D")

if __name__=="__main__":build()
