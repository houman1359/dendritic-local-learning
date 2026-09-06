#!/usr/bin/env python3
"""Selected framework and image-task reference panels for credit-first Fig. 1.

Reuse the public eligibility and MNIST drawing functions; all new contrasts
read frozen release tables. No fitting or mutation of earlier figure inputs.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
import build_main_figure_01 as framework
import build_main_figure_02 as image_reference
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF,
                           MARKER_MS, PT_ANNOT, PT_LABEL, PT_LEGEND, PT_SMALL,
                           Margins, NativeCanvas, style_panel)
from journal_style import style_direct_color_labels
from native_schematics import Frame

SOURCE = JOURNAL / "source_data"
OUT = JOURNAL / "figures/components/credit_first_figure_01.pdf"
RECORDS = SOURCE / "credit_first_figures"


def dictionaries(ax):
    """Illustrative six-site tree and three exact delivery matrices."""
    f = Frame(ax)
    ax.text(.50,.94,"Within one neuron: delivered field = A c",ha="center",fontsize=PT_LABEL)
    x_positions = [.12,.45,.79]
    names = ["Broadcast", "Subtrees", "Resolved"]
    matrices = [np.ones((6,1)), np.array([[1,0],[0,1],[1,0],[1,0],[0,1],[0,1]]), np.eye(6)]
    tones = [COLORS["additive"],COLORS["shunting"],COLORS["bp"]]
    for cx, name, matrix, tone in zip(x_positions,names,matrices,tones):
        # Public six-site morphology glyph: two proximal and four distal sites.
        framework.mini_tree(f,cx+.05,.71,.12)
        ax.text(cx,.505,name,ha="center",fontsize=PT_ANNOT,color=tone)
        width = [0,.045,.10,0,0,0,.205][matrix.shape[1]]
        ia = ax.inset_axes([cx-width/2,.14,width,.28])
        cmap=LinearSegmentedColormap.from_list("dictionary",["white",tone])
        ia.imshow(matrix,aspect="auto",interpolation="nearest",cmap=cmap,vmin=0,vmax=1)
        ia.set_xticks([]);ia.set_yticks([])
        for spine in ia.spines.values():spine.set_color(COLORS["edge"]);spine.set_linewidth(LW_HAIR)
        ax.text(cx,.055,f"K = {matrix.shape[1]}",ha="center",fontsize=PT_ANNOT)
    ax.text(.50,-.045,"Rows: 6 compartments; columns: spatial profiles",ha="center",fontsize=PT_SMALL,color=COLORS["mute"])
    return matrices


def between_neurons(ax):
    f = Frame(ax)
    f.disc((.075,.59),3.6,fill=COLORS["bp"])
    ax.text(.075,.77,"output\nerror",ha="center",va="center",fontsize=PT_ANNOT)
    for y,text in [(.79,"Readout derivative"),(.37,"Fixed random map\n(DFA)")]:
        box=(.24,y-.14,.47,.28)
        f.group(box,tint=COLORS["panel_bg"],edge=COLORS["edge"],lw=LW_EDGE,radius_pt=2)
        ax.text(.475,y,text,ha="center",va="center",fontsize=PT_LABEL)
        f.arrow((.105,.59),(.23,y),color=COLORS["bp"],lw=LW_EDGE,head=3.5)
        f.arrow((.72,y),(.84,y),color=COLORS["additive"],lw=LW_EDGE,head=3.5)
        ax.text(.91,y,"δ₁\nδ₂\n…",ha="center",va="center",fontsize=PT_LABEL,color=COLORS["additive"])
    ax.text(.90,.95,"neurons",ha="center",fontsize=PT_ANNOT)
    ax.text(.025,.045,"Then: retain neuron identity or share one scalar",ha="left",fontsize=PT_ANNOT)
    ax.text(.025,-.075,"Within each tree: broadcast or exact transport",ha="left",fontsize=PT_ANNOT,color=COLORS["mute"])


def contrasts():
    ladder=pd.read_csv(SOURCE / "mnist_feedback_ladder/paired_contrasts.csv")
    factorial=pd.read_csv(SOURCE / "mnist_between_within_factorial/paired_contrasts.csv")
    cifar=pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv")
    rows=[]
    for endpoint,readout_name,dfa_name,cifar_name in [
        ("neuron_identity","neuron specific - scalar broadcast","dfa within: neuron - scalar","neuron specific minus strict scalar"),
        ("path_resolution","exact path - neuron specific","dfa within: exact path - neuron","exact path minus neuron specific")]:
        for feedback, table, key in [("Readout",ladder,readout_name),("DFA",factorial,dfa_name)]:
            for architecture in ["shunting","additive"]:
                row=table[table.architecture.eq(architecture)&table.contrast.eq(key)].iloc[0]
                rows.append(dict(endpoint=endpoint,task="MNIST",source=feedback,architecture=architecture,
                                 label=f"{feedback}: {'shunt' if architecture=='shunting' else 'additive'}",
                                 mean_pp=100*row.mean_difference,low_pp=100*row.ci95_low,high_pp=100*row.ci95_high,
                                 n_seeds=int(row.n_seeds),source_table=str((SOURCE / ("mnist_feedback_ladder/paired_contrasts.csv" if feedback=="Readout" else "mnist_between_within_factorial/paired_contrasts.csv")).relative_to(JOURNAL)),
                                 source_contrast=key))
        row=cifar[cifar.contrast.eq(cifar_name)].iloc[0]
        rows.append(dict(endpoint=endpoint,task="CIFAR-10",source="Readout",architecture="additive",label="CIFAR: additive",
                         mean_pp=100*row.mean_difference,low_pp=100*row.ci95_low_difference,high_pp=100*row.ci95_high_difference,
                         n_seeds=int(row.n_seeds),source_table="source_data/cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv",source_contrast=cifar_name))
    return pd.DataFrame(rows)


def forest(ax, rows, *, path=False):
    ax.axvline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    for y,row in zip([4,3,2,1,-.3],rows.itertuples()):
        color=COLORS["shunting" if row.architecture=="shunting" else "additive"]
        marker="o" if row.architecture=="shunting" else "s"
        ax.errorbar(row.mean_pp,y,xerr=[[row.mean_pp-row.low_pp],[row.high_pp-row.mean_pp]],
                    fmt=marker,color=color,ms=MARKER_MS,elinewidth=LW_ERR,capsize=2,
                    markerfacecolor="white",markeredgewidth=LW_ERR)
    ax.set_yticks([4,3,2,1,-.3],rows.label)
    ax.set_ylim(-.9,4.7)
    ax.axhline(.35,color=COLORS["grid"],lw=LW_HAIR)
    if path:
        ax.set_xlim(-1.32,.39);ax.set_xticks([-1,-.5,0])
        ax.set_xlabel("Exact path − neuron-specific (pp)")
    else:
        ax.set_xlim(-.7,18.1);ax.set_xticks([0,5,10,15])
        ax.set_xlabel("Neuron-specific − scalar (pp)")
    style_panel(ax,grid="x")
    ax.tick_params(axis="y",length=0,labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)


def main():
    RECORDS.mkdir(exist_ok=True)
    canvas=NativeCanvas(534/72,3,row_weights=[126,127,145],hgutter_pt=38,vgutter_pt=44,
                        margins=Margins(left=36,right=12,top=24,bottom=42))
    a=canvas.panel("A",0,0,6,schematic=True,title="Eligibility × delivered credit",lock=False)
    b=canvas.panel("B",0,6,6,schematic=True,title="One tree, three dictionaries",lock=False)
    c=canvas.panel("C",1,0,6,schematic=True,title="Feedback between neurons",lock=False)
    d=canvas.panel("D",1,6,6,title="MNIST test accuracy")
    e=canvas.panel("E",2,0,6,title="Preserving neuron identity")
    f=canvas.panel("F",2,6,6,title="Adding exact path resolution")
    framework.panel_factorization_readable(a)
    matrices=dictionaries(b)
    between_neurons(c)
    image_reference.panel_mnist_ladder(d)
    d.set_xticklabels(["Layer\nscalar","Per\nneuron","Exact\npath"])
    for label in d.get_xticklabels():
        label.set_rotation(0)
        label.set_ha("center")
    rows=contrasts()
    forest(e,rows[rows.endpoint.eq("neuron_identity")])
    forest(f,rows[rows.endpoint.eq("path_resolution")],path=True)
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    # NativeCanvas aligns the leading letters; extend that same alignment to
    # this sheet's second column after all forest labels have been measured.
    right_letters=[item["art"] for item in canvas._letters if item["letter"] in "BDF"]
    shared_x=min(art.get_position()[0] for art in right_letters)
    for art in right_letters:
        art.set_position((shared_x,art.get_position()[1]))
    problems=canvas.save(OUT,name="credit_first_figure_01",dpi=180,lock=False)
    rows.to_csv(RECORDS / "figure_01_contrasts.csv",index=False)
    np.savez_compressed(RECORDS / "figure_01_illustrative_dictionaries.npz",broadcast=matrices[0],subtrees=matrices[1],resolved=matrices[2])
    files=[Path(__file__),Path(framework.__file__),Path(image_reference.__file__),
           JOURNAL / "scripts/figure_canvas.py",JOURNAL / "scripts/journal_style.py",
           SOURCE / "mnist_feedback_ladder/condition_summary.csv",SOURCE / "mnist_feedback_ladder/seed_outcomes.csv",
           SOURCE / "mnist_feedback_ladder/paired_contrasts.csv",SOURCE / "mnist_between_within_factorial/paired_contrasts.csv",
           SOURCE / "cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv"]
    mapping={"A":"Public build_main_figure_01.panel_factorization_readable; directed-tree eligibility and exact path example",
             "B":"New six-site illustrative matrices; no experimental data or fitted coefficients",
             "C":"New diagram of readout-derivative versus fixed-random feedback source; does not equate DFA with within-tree broadcast",
             "D":"Public build_main_figure_02.panel_mnist_ladder; original 15-seed MNIST means, intervals and paired points",
             "E":"Original paired neuron-minus-scalar contrasts, two MNIST sources plus flattened CIFAR-10 readout source",
             "F":"Original exact-path-minus-neuron contrasts, same row ordering and sources as E"}
    payload=dict(panel_sources=mapping,source_sha256={str(p.relative_to(JOURNAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                 numerical_scope="No new fitting, bootstrap, significance test or modification to source outcomes; existing confidence intervals replayed exactly.",
                 layout_findings=problems)
    (RECORDS / "figure_01_sources.json").write_text(json.dumps(payload,indent=2)+"\n")
    caption=("**Task-derived credit separates neuron identity from resolution within a dendritic tree.** "
       "**A**, Synaptic conductance changes combine a local eligibility with delivered compartment credit. "
       "Here η is the learning rate; xᵢ is presynaptic activity; Rₙ is total input resistance; Eᵢ−Vₙ is the "
       "synaptic driving force; and εₙ is the delivered voltage-error signal. In a directed tree, exact "
       "transport multiplies somatic error δ₀ by local path derivatives α. Reciprocal cables require "
       "an adjoint solution. **B**, An illustrative neuron with six nonsomatic compartments supports an "
       "all-site broadcast, two subtree profiles, or six independent compartment profiles. The delivery "
       "matrix A has one row per compartment and one column per spatial profile; c supplies their "
       "example-dependent coefficients, and K counts profiles. These are possible field representations, "
       "not six independently supplied external errors: exact paths can transform one somatic error "
       "using state-dependent local derivatives. **C**, Neuronal coordinates are supplied either by the "
       "task-readout derivative or a fixed random output-to-neuron map, termed direct feedback alignment "
       "(DFA). The choice between neurons is crossed with the resolution used within each tree. The "
       "layer-scalar control reduces neuronal coordinates to a shared scalar as defined in Methods; "
       "the per-neuron condition broadcasts each neuron's own coordinate through its arbor. **D**, "
       "MNIST test accuracy with readout-derived coordinates in shunting and raw-additive trees. Thin "
       "lines join paired seeds; large symbols and intervals show means and retained 95% confidence "
       "intervals (15 seeds per architecture). **E–F**, Paired accuracy differences for preserving neuron "
       "identity (E) and then adding exact path resolution (F), using readout or DFA sources on MNIST "
       "and a readout source on flattened CIFAR-10. Colors identify architectures as in D; 'additive' "
       "denotes the raw-additive comparator. CIFAR-10 uses 20 seeds. Existing confidence intervals "
       "are reproduced without recomputation; pp denotes percentage points. MNIST's raw-additive "
       "path increment is small and positive, whereas CIFAR-10's is negative, bounding the claim to "
       "the predominance of neuron identity in these image tasks.\n")
    (RECORDS / "figure_01_caption.md").write_text(caption)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()
