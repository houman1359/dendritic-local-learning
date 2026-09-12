#!/usr/bin/env python3
"""Second-animal routing results with explicit candidate-size diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_REF, PT_ANNOT,
                           PT_SMALL, Margins, NativeCanvas)
from analyze_pinky_v185_replication import METHOD_ORDER, bootstrap_mean
from analyze_pinky_v185_replication import METHOD_STYLE as _PINKY_STYLE

ROOT=Path(__file__).resolve().parents[1]
# Ancestry routes keep the anatomy sheets' green (S10 B and S20 B draw the
# same arm in COLORS["shunting"]); the neutral "dend" grey of the shared
# style would read as a second grey beside the random-path control.
METHOD_STYLE={m:((COLORS["shunting"] if m=="morphology-aware paths" else c),ls,mk)
              for m,(c,ls,mk) in _PINKY_STYLE.items()}
SOURCE=ROOT/"source_data/pinky_v185_replication"
OUT=ROOT/"figures/supplementary/figure_S27_panels_A-C.pdf"


def build():
    cohort=pd.read_csv(SOURCE/"cohort_manifest.csv")
    allcells=pd.read_csv(SOURCE/"routing/cell_metrics.csv")
    cells=allcells[allcells.qc_included]
    curves=pd.read_csv(SOURCE/"routing/feedback_compression_curves.csv")
    cv=NativeCanvas(395/72,2,hgutter_pt=32,vgutter_pt=62,
                    margins=Margins(left=46,right=14,top=20,bottom=38))
    a=cv.panel("A",0,0,6,title="Second mouse: x–y projection")
    b=cv.panel("B",0,6,6,title="Residual versus route budget",grid="y")
    c=cv.panel("C",1,0,6,title="Residual reduction per coefficient",grid="y")
    d=cv.panel("D",1,6,6,title="Candidate routes and field rank",grid="y")
    colors=np.where(cohort.root_id.isin(cells.root_id),COLORS["dend"],COLORS["mute"])
    a.scatter(cohort.x_nm/1000,cohort.y_nm/1000,c=colors,s=20)
    for row in cohort.itertuples():
        # Cell 11 is labelled on its left: its right side touches cell 12.
        left=row.selection_order+1==11
        a.text(row.x_nm/1000+(-1.6 if left else 1.2),row.y_nm/1000,str(row.selection_order+1),
               fontsize=PT_SMALL,ha="right" if left else "left")
    a.set_xlabel("volume x (µm)");a.set_ylabel("volume y (µm)");a.set_ylim(150,315)
    a.text(.02,.96,f"{len(cells)} qualifying / {len(cohort)} selected trees",transform=a.transAxes,
            va="top",fontsize=PT_SMALL,color=COLORS["mute"])
    # Every qualifying cell is drawn behind its arm's mean (faint, offset a
    # little along the log axis per arm), so the cell-to-cell spread that the
    # s.e.m. bar does not show is on the panel.
    for k,method in enumerate(METHOD_ORDER):
        part=curves[curves.method.eq(method)]
        frame=part.groupby("channels").residual.agg(["mean","sem"])
        color,ls,marker=METHOD_STYLE[method]
        shift=2**((k-2)*0.055)
        b.scatter(part.channels*shift,part.residual,s=6,color=color,alpha=.30,linewidths=0,zorder=1)
        b.errorbar(frame.index,frame["mean"],yerr=frame["sem"].fillna(0),color=color,
                    ls=ls,marker=marker,ms=3.5,lw=LW_DATA,capsize=2,zorder=3,
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
    # Neutral markers: green belongs to the ancestry arm on this sheet.  The
    # two selected cells that failed the typed-input criterion are drawn open,
    # so the cohort numbering 1-12 is complete on the panel.
    excluded=allcells[~allcells.qc_included]
    d.scatter(cells.n_i_segments,cells.credit_kernel_participation_rank,s=22,color=COLORS["mute"],zorder=3)
    d.scatter(excluded.n_i_segments,excluded.credit_kernel_participation_rank,s=22,
              facecolors="white",edgecolors=COLORS["mute"],linewidths=LW_EDGE,zorder=3)
    # Label offsets in points; cell 11 sits against cell 4, so it goes left.
    offsets={11:(-13,1),4:(3,-8),1:(3,-8)}
    for row in allcells.itertuples():
        label=row.selection_order+1
        d.annotate(str(label),(row.n_i_segments,row.credit_kernel_participation_rank),
                    xytext=offsets.get(label,(3,2)),textcoords="offset points",fontsize=PT_SMALL,
                    color=COLORS["ink"] if row.qc_included else COLORS["mute"])
    d.set_xlabel("inhibitory-bearing segments (candidate routes)")
    d.set_xticks([4,8,12,16,20]);d.set_xlim(2.5,22.5)
    d.set_ylabel("kernel participation rank");d.set_ylim(0.85,2.85)
    return cv.save(OUT,name="figure_S27_panels_A-D")

if __name__=="__main__":build()
