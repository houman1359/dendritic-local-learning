#!/usr/bin/env python3
"""Task coefficients and matched-bandwidth ancestry evidence for main Fig. 3."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
JOURNAL=HERE.parents[1]
sys.path.insert(0,str(JOURNAL / "scripts"))
import build_main_figure_05 as original
import routing_figure_panels as routing
import run_trained_subtree_address_full_factorial as experiment
from figure_canvas import (COLORS,LW_DATA,LW_EDGE,LW_ERR,LW_HAIR,LW_REF,MARKER_MS,
                           PT_ANNOT,PT_LABEL,PT_SMALL,Margins,NativeCanvas,style_panel)
from journal_style import style_direct_color_labels

SOURCE=JOURNAL / "source_data"
DATA=SOURCE / "trained_subtree_address_full_factorial"
REVIEW=SOURCE / "review_evidence_reanalysis"
RECORDS=SOURCE / "credit_first_figures"
CONFIG=JOURNAL / "configs/trained_subtree_address/full_factorial_confirmatory.json"
OUT=JOURNAL / "figures/components/credit_first_figure_03.pdf"


def coefficient_prediction():
    cfg=json.loads(CONFIG.read_text())["task"]
    n=cfg["contexts"]
    rows=[]
    for k in [1,2,4,8]:
        routes=experiment.grouped_routes(np.arange(n),k,"correct_ancestry_subtrees",np.random.default_rng(0))
        for context in range(n):
            amplitudes=np.array([cfg["selected_signal"] if stream==context else
                -cfg["distractor_signal_by_tree_distance"][experiment.tree_relation(stream,context)] for stream in range(n)])
            mask=routes[context]>0
            raw=float(amplitudes[mask].sum())
            delivered=float(routes[context]@amplitudes)
            np.testing.assert_allclose(delivered,raw/np.sqrt(mask.sum()),atol=1e-14)
            rows.append(dict(budget_k=k,context=context,group_size=int(mask.sum()),raw_coefficient_sum=raw,
                             normalized_coefficient_sum=delivered,normalization="unit Euclidean norm per route row"))
    table=pd.DataFrame(rows)
    assert table.groupby("budget_k").raw_coefficient_sum.std().max()<1e-14
    np.testing.assert_allclose(table.groupby("budget_k").raw_coefficient_sum.mean(),[-3.05,-.05,.85,1.0],atol=1e-14)
    return table


def prediction_panel(ax,table):
    values=table.groupby("budget_k").raw_coefficient_sum.mean().to_numpy()
    ax.axhline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    ax.plot(range(4),values,marker="o",color=COLORS["shunting"],lw=LW_DATA,ms=MARKER_MS)
    for x,y in enumerate(values):
        ax.text(x,y+(.29 if x!=1 else -.5),f"{y:+.2f}".replace("-","−"),ha="center",va="center",fontsize=PT_ANNOT)
    ax.set_xticks(range(4),["1","2","4","8"])
    ax.set_xlim(-.42,3.45);ax.set_ylim(-3.7,1.65);ax.set_yticks([-3,-2,-1,0,1])
    ax.set_ylabel("Raw coefficient sum")
    ax.set_xlabel("Channels, K")
    style_panel(ax)


def primary_effect(ax,summary,pairs):
    key="best_matched_nonanatomical_oracle"
    row=summary[summary.control.eq(key)].iloc[0]
    values=pairs[pairs.control.eq(key)].accuracy_difference_pp.to_numpy()
    ax.scatter(values,.27+np.linspace(-.14,.14,len(values)),s=10,color=COLORS["oracle"],alpha=.4,edgecolors="none")
    ax.errorbar(row.mean_difference_pp,.27,xerr=[[row.mean_difference_pp-row.ci95_low_pp],[row.ci95_high_pp-row.mean_difference_pp]],
                fmt="D",color=COLORS["oracle"],mfc="white",ms=MARKER_MS+1,elinewidth=LW_ERR,capsize=2,zorder=4)
    ax.axvline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    ax.text(.99,.91,f"{row.mean_difference_pp:+.2f} pp [{row.ci95_low_pp:.2f}, {row.ci95_high_pp:.2f}]   "
            f"Holm–Wilcoxon P = {row.p_holm_four_budgets:.4f}",transform=ax.transAxes,ha="right",va="top",fontsize=PT_ANNOT)
    ax.text(.01,.91,f"{int(row.positive_seeds)}/20 paired seeds positive",transform=ax.transAxes,ha="left",va="top",fontsize=PT_ANNOT)
    ax.set_xlim(-1.9,4.35);ax.set_xticks([-1,0,1,2,3,4]);ax.set_ylim(-.15,1.10);ax.set_yticks([])
    ax.set_xlabel("Ancestry − strongest matched control (percentage points)")
    style_panel(ax,grid="x");ax.spines["left"].set_visible(False)


def individual_effects(ax,summary,pairs):
    selected=[("random_rank_k","Dense rank-4"),("random_sparse_matched","Random sparse"),
              ("depth_interleaved_bins","Depth-interleaved")]
    for y,(key,label) in zip([2,1,0],selected):
        row=summary[summary.control.eq(key)].iloc[0]
        values=pairs[pairs.control.eq(key)].accuracy_difference_pp.to_numpy()
        ax.scatter(values,y+np.linspace(-.14,.14,len(values)),s=9,color=COLORS["oracle"],alpha=.33,edgecolors="none")
        ax.errorbar(row.mean_difference_pp,y,xerr=[[row.mean_difference_pp-row.ci95_low_pp],[row.ci95_high_pp-row.mean_difference_pp]],
                    fmt="D",color=COLORS["oracle"],mfc="white",ms=MARKER_MS,elinewidth=LW_ERR,capsize=2,zorder=4)
    ax.axvline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    ax.set_yticks([2,1,0],[x[1] for x in selected]);ax.set_ylim(-.5,2.5)
    ax.set_xlim(-2.2,14.1);ax.set_xticks([0,4,8,12])
    ax.set_xlabel("Ancestry − individual control (percentage points)")
    style_panel(ax,grid="x");ax.tick_params(axis="y",length=0,labelsize=PT_ANNOT)
    ax.spines["left"].set_visible(False)


def main():
    RECORDS.mkdir(exist_ok=True)
    outcomes=pd.read_csv(DATA / "seed_outcomes.csv")
    summary=pd.read_csv(DATA / "condition_summary.csv")
    contrasts=pd.read_csv(REVIEW / "ancestry_k4_control_contrasts.csv")
    pairs=pd.read_csv(REVIEW / "ancestry_control_paired_differences.csv")
    pairs=pairs[pairs.budget_k.eq(4)]
    prediction=coefficient_prediction()
    canvas=NativeCanvas(556/72,4,row_weights=[128,104,65,99],hgutter_pt=38,vgutter_pt=33,
                        margins=Margins(left=39,right=13,top=24,bottom=34))
    a=canvas.panel("A",0,0,7,title="Eight-context hierarchical task",schematic=True,lock=False)
    b=canvas.panel("B",0,7,5,title="Coefficient sums set by task design")
    c=canvas.panel("C",1,0,6,title="Learning across channel budgets",lock=False)
    d=canvas.panel("D",1,6,6,title="Matching task groups to the tree",lock=False)
    e=canvas.panel("E",2,0,12,title="K = 4: ancestry exceeds the strongest matched control",lock=False)
    f=canvas.panel("F",3,0,12,title="K = 4: ancestry versus individual controls")
    original.hierarchical_task(a)
    prediction_panel(b,prediction)
    routing.bandwidth_sweep_compact(c,outcomes,summary[summary.architecture.eq("dendritic_tree")])
    c.axhline(.5,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    c.set_ylabel("Held-out accuracy")
    for text in c.texts:
        if text.get_text()=="best control":text.set_text("best matched control");text.set_position((.03,.70))
        elif text.get_text()=="correct":text.set_text("ancestry");text.set_position((1.53,.843))
    c.text(3.12,.505,"chance",ha="right",va="bottom",fontsize=PT_SMALL,color=COLORS["mute"])
    routing.topology_alignment_compact(d,outcomes)
    d.set_ylabel("Matched − rewired accuracy (pp)")
    d.text(.98,.92,"K = 1 and 8: exact ties",transform=d.transAxes,ha="right",va="top",fontsize=PT_SMALL,color=COLORS["mute"])
    primary_effect(e,contrasts,pairs)
    individual_effects(f,contrasts,pairs)
    canvas.declare_reserve("F",left=80)
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    right_letters=[item["art"] for item in canvas._letters if item["letter"] in "BD"]
    shared_x=min(art.get_position()[0] for art in right_letters)
    for art in right_letters:art.set_position((shared_x,art.get_position()[1]))
    problems=canvas.save(OUT,name="credit_first_figure_03",dpi=180,lock=False)
    prediction.to_csv(RECORDS / "figure_03_coefficient_prediction.csv",index=False)
    contrasts[contrasts.control.isin(["best_matched_nonanatomical_oracle","random_rank_k","random_sparse_matched","depth_interleaved_bins"])].to_csv(RECORDS / "figure_03_k4_contrasts.csv",index=False)
    files=[Path(__file__),Path(original.__file__),Path(routing.__file__),Path(experiment.__file__),CONFIG,
           JOURNAL / "scripts/figure_canvas.py",JOURNAL / "scripts/journal_style.py",
           DATA / "seed_outcomes.csv",DATA / "condition_summary.csv",REVIEW / "ancestry_k4_control_contrasts.csv",
           REVIEW / "ancestry_control_paired_differences.csv"]
    mapping={"A":"Public build_main_figure_05.hierarchical_task; selected stream +1 and fixed negative distractor coefficients",
             "B":"Algebraic sum from the frozen task configuration and the actual grouped_routes support, verified across all 8 contexts; no new prospective experiment",
             "C":"Public routing_figure_panels.bandwidth_sweep_compact; original 20-seed accuracies and original mean/interval functions",
             "D":"Public routing_figure_panels.topology_alignment_compact; original paired matched-versus-rewired statistics and resampling seeds",
             "E":"Frozen ancestry_k4_control_contrasts primary maximum-over-four comparator, paired points and Holm adjustment across four K budgets",
             "F":"Same frozen reanalysis: dense rank-4, random sparse and depth-interleaved contrasts; large derangement effect remains visible in C rather than setting F's scale"}
    payload=dict(panel_sources=mapping,source_sha256={str(p.relative_to(JOURNAL)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                 layout_findings=problems,coefficient_scope="Raw group sums; implemented route rows divide by sqrt(group size). Signs follow from the generator and are not a newly prospective accuracy prediction.")
    (RECORDS / "figure_03_sources.json").write_text(json.dumps(payload,indent=2)+"\n")
    caption=("**Hierarchical distractors expose a benefit of ancestry routing at intermediate bandwidth.** "
       "**A**, Eight streams carry a binary label sign sᵢ along stream-specific teacher features, with feature noise. "
       "The cue selects a +sᵢ stream; its sibling, other same-half streams and opposite-half streams carry "
       "−0.15sᵢ, −0.45sᵢ and −0.75sᵢ, respectively. The tree highlights the selected stream's ancestry. "
       "**B**, Summing label coefficients within the cued ancestry group gives −3.05, −0.05, 0.85 and 1.00 "
       "for K=1, 2, 4 and 8 channels. These are unnormalized sums fixed by the task design and verified "
       "for all eight contexts. Implemented feedback rows divide by the square root of group size, preserving "
       "the sign change at K=4. This is an algebraic design consequence, not a newly prospective prediction "
       "of trained accuracy; at K=8, terminal-stream resolution removes the grouping distinction. "
       "**C**, Held-out accuracy across K for correct ancestry, route derangement, and the per-seed maximum "
       "of four matched controls (dense rank-K, random sparse, depth-interleaved and deranged routes). "
       "Points and intervals reproduce 20-seed means and 95% seed-bootstrap intervals; the dashed line is "
       "chance. The ancestry/derangement tie at K=1 and ancestry/best-control tie at K=8 are exact by "
       "construction. **D**, Paired accuracy difference between task-matched and degree/depth-matched "
       "rewired trees under correct ancestry feedback. K=1 and K=8 give exact ties. **E**, At K=4, "
       "ancestry exceeds the strongest matched comparator by 1.27 percentage points (95% CI 0.59–1.95), "
       "with 15/20 paired seeds positive. The displayed two-sided Wilcoxon P=0.0101 uses Holm adjustment "
       "across four channel budgets. **F**, The individual dense, sparse and depth-interleaved contrasts "
       "are displayed on their own effect scale; the much larger derangement contrast does not set that "
       "axis range. In E–F, small points are paired seeds, diamonds are means and bars are the frozen "
       "95% intervals. The comparator maximum uses held-out outcomes as an assessment ceiling, not "
       "as a learned biological selector.\n")
    (RECORDS / "figure_03_caption.md").write_text(caption)
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
