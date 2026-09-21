#!/usr/bin/env python3
"""Native publication figure for the frozen opponent-conductance comparison.

No outcomes are fitted or selected here. Primary and extended outcomes retain
their common paired seed blocks. The three-profile rule uses oracle projection
coefficients; binary context generates at most two current path profiles.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import FixedLocator, FixedFormatter

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
ROOT = JOURNAL / "source_data/conductance_credit_demand/opponent"
SOURCE = ROOT / "summaries"
DEST = ROOT / "figures"
sys.path.insert(0, str(HERE.parent))
from figure_canvas import (
    NativeCanvas, Margins, COLORS, PT_LABEL, PT_ANNOT, PT_SMALL, PT_LEGEND,
    LW_DATA, LW_REF, LW_EDGE, LW_ERR, LW_HAIR, MARKER_MS, ERR_CAPSIZE,
)
from journal_style import style_direct_color_labels

RULES = ["exact", "unit_broadcast", "calibrated_broadcast", "ancestry_three_oracle"]
COLOR = dict(exact=COLORS["bp"], unit_broadcast=COLORS["scalar"],
             calibrated_broadcast=COLORS["additive"], ancestry_three_oracle=COLORS["oracle"])
STYLE = dict(exact="-", unit_broadcast="--", calibrated_broadcast=":", ancestry_three_oracle="-.")
TASKS = ["aligned_strong", "opposed_strong"]
EXPECTED_SEEDS = set(range(2101, 2121))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values):
    values = np.asarray(values, float)
    assert len(values) == 20 and np.isfinite(values).all()
    draws = values[np.random.default_rng(982211).integers(len(values), size=(20000, len(values)))].mean(axis=1)
    low, high = np.quantile(draws, [.025, .975])
    return dict(mean=float(values.mean()), ci_low=float(low), ci_high=float(high), n=20)


def circuit(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(-.16, 1.01)
    positions = {0:(.5,.93), 1:(.25,.64), 2:(.75,.64),
                 3:(.05,.33), 4:(.36,.33), 5:(.64,.33), 6:(.95,.33)}
    for child in range(1, 7):
        parent = (child-1)//2
        ax.annotate("", xy=positions[parent], xytext=positions[child],
                    arrowprops=dict(arrowstyle="-|>", mutation_scale=6,
                                    color=COLORS["dend"], lw=LW_EDGE, shrinkA=4, shrinkB=4), zorder=1)
    for node, (x,y) in positions.items():
        ax.plot(x,y,"o",ms=6.3 if node==0 else 5.5,
                mfc=COLORS["soma"] if node==0 else "white", mec=COLORS["dend"], mew=LW_EDGE, zorder=3)
    ax.text(.56,.97,"somatic output",fontsize=PT_ANNOT,va="center")
    for parent,label in [(1,"c = 1"),(2,"c = 0")]:
        x,y=positions[parent]
        ax.plot([x-.14,x-.035],[y,y],color=COLORS["inh"],lw=LW_DATA,zorder=4)
        ax.plot([x-.035,x-.035],[y-.04,y+.04],color=COLORS["inh"],lw=LW_DATA,zorder=4)
        ax.text(x-.10,y+.12,label,color=COLORS["inh"],ha="center",fontsize=PT_ANNOT)
    for node,label in zip(range(3,7),["z₁","z₂","z₁","z₂"]):
        x,y=positions[node]
        ax.text(x,y-.13,label,ha="center",fontsize=PT_LABEL)
    ax.text(.5,.015,"Each leaf receives two E and two I channels",ha="center",fontsize=PT_SMALL)
    ax.text(.5,-.14,"24 trainable conductances; one somatic error",ha="center",fontsize=PT_SMALL)


def tuning(ax):
    """Nominal teacher leaf curves: actual E/(1+E+I), with no invented data."""
    ax.set_xlim(-.02,1.03)
    ax.set_ylim(-.20,1.03)
    ax.text(.46,.94,"Left subtree",ha="center",fontsize=PT_ANNOT)
    ax.text(.82,.94,"Right subtree",ha="center",fontsize=PT_ANNOT)
    z=np.linspace(-2,2,151)
    table=[]
    for row,label,y0 in [(0,"Aligned",.57),(1,"Opposed",.13)]:
        ax.text(.0,y0+.10,label,fontsize=PT_LABEL,ha="left",va="center")
        for col,center in enumerate([.46,.82]):
            opposed = row==1 and col==1
            gp,gm=(.25,8.) if opposed else (8.,.25)
            e=gp*np.exp(z)+gm*np.exp(-z)
            inhibitory=gm*np.exp(z)+gp*np.exp(-z)
            voltage=e/(1+e+inhibitory)
            xx=center+.135*z/2
            yy=y0+.22*voltage
            ax.plot([center-.15,center+.15],[y0,y0],color=COLORS["mute"],lw=LW_HAIR)
            ax.plot([center-.15,center-.15],[y0,y0+.23],color=COLORS["mute"],lw=LW_HAIR)
            ax.plot(xx,yy,color=COLORS["dend"],lw=LW_DATA)
            for zi,vi in zip(z,voltage):
                table.append(dict(panel="B",task=label.lower(),subtree="left" if col==0 else "right",latent=zi,nominal_leaf_voltage=vi))
    ax.text(.5,-.055,"Leaf voltage versus latent input z",ha="center",fontsize=PT_SMALL)
    ax.text(.5,-.18,"Identical input samples in the two tasks",ha="center",fontsize=PT_SMALL)
    return table


def curves_panel(ax,task,curves,table,legend=False):
    handles={}
    maximum_control_difference=0.
    for rule in RULES:
        part=curves[curves.task.eq(task)&curves.rule.eq(rule)]
        rows=[]
        for step,group in part.groupby("step",sort=True):
            group=group.sort_values("seed")
            assert set(group.seed)==EXPECTED_SEEDS and len(group)==20
            row=dict(step=int(step),**interval(group.test_nmse))
            rows.append(row)
            table.append(dict(panel="C" if task==TASKS[0] else "D",task=task,rule=rule,**row))
        frame=pd.DataFrame(rows)
        ax.fill_between(frame.step,frame.ci_low,frame.ci_high,color=COLOR[rule],alpha=.11,lw=0,zorder=1)
        line,=ax.plot(frame.step,frame["mean"],color=COLOR[rule],ls=STYLE[rule],lw=LW_DATA,zorder=3)
        handles[rule]=line
    pair=curves[curves.task.eq(task)&curves.rule.isin(["unit_broadcast","calibrated_broadcast"])].pivot(
        index=["seed","step"],columns="rule",values="test_nmse")
    maximum_control_difference=float(np.max(np.abs(pair.unit_broadcast-pair.calibrated_broadcast)))
    ax.axvline(4096,color=COLORS["mute"],ls=":",lw=LW_REF,zorder=0)
    ax.set_xlim(0,16800)
    ax.set_yscale("log")
    ax.set_ylim(1e-7,3)
    ax.set_xticks([0,4096,16384],["0","4,096","16,384"])
    ax.yaxis.set_major_locator(FixedLocator([1e-6,1e-4,1e-2,1]))
    ax.yaxis.set_major_formatter(FixedFormatter(["10⁻⁶","10⁻⁴","10⁻²","1"]))
    ax.minorticks_off()
    ax.set_xlabel("Training updates")
    ax.set_ylabel("Test NMSE" if legend else "")
    if not legend:
        ax.set_yticklabels([])
    if legend:
        hs=[handles["exact"],handles["ancestry_three_oracle"],
            (handles["unit_broadcast"],handles["calibrated_broadcast"])]
        labels=["Exact path","Three profiles (oracle)","Unit / calibrated broadcast"]
        ax.legend(hs,labels,handler_map={tuple:HandlerTuple(ndivide=None)},loc="upper right",
                  frameon=False,fontsize=PT_SMALL,handlelength=2.2,borderaxespad=.2,labelspacing=.25)
    return maximum_control_difference


def contrasts_panel(ax,contrasts,seeds,table):
    specifications=[("fresh","adam","Adam\n4,096"), ("extension","adam","Adam\n16,384"),
                    ("fresh","sgd","SGD\n4,096"), ("extension","sgd","SGD\n16,384")]
    for y,(phase,optimizer,label) in enumerate(specifications):
        mask=contrasts.phase.eq(phase)&contrasts.optimizer.eq(optimizer)&contrasts.rule.eq("calibrated_broadcast")&contrasts.contrast.eq("task_difference_in_gap")
        row=contrasts[mask&contrasts.rate_scope.eq("selected")].iloc[0]
        common=contrasts[mask&contrasts.rate_scope.eq("common")].iloc[0]
        np.testing.assert_allclose(row[["mean","ci_low","ci_high"]].to_numpy(float),common[["mean","ci_low","ci_high"]].to_numpy(float),atol=0,rtol=0)
        part=seeds[seeds.phase.eq(phase)&seeds.optimizer.eq(optimizer)&seeds.rule.eq("calibrated_broadcast")&seeds.contrast.eq("task_difference_in_gap")&seeds.rate_scope.eq("selected")].sort_values("seed")
        assert set(part.seed)==EXPECTED_SEEDS and len(part)==20
        ax.scatter(part.value,y+np.linspace(-.15,.15,20),s=8,color=COLORS["mute"],alpha=.4,linewidths=0,zorder=2)
        ax.errorbar(row["mean"],y,xerr=[[row["mean"]-row.ci_low],[row.ci_high-row["mean"]]],
                    fmt="o" if phase=="fresh" else "s",mfc="white" if phase=="fresh" else COLORS["bp"],
                    color=COLORS["bp"],ms=MARKER_MS,lw=LW_ERR,capsize=ERR_CAPSIZE,zorder=4)
        table.append(dict(panel="E",**row.to_dict()))
    ax.axvline(0,color=COLORS["mute"],ls=":",lw=LW_REF)
    ax.set_xlim(-.04,1.05)
    ax.set_ylim(3.55,-.85)
    ax.set_yticks(range(4),[v[2] for v in specifications],fontsize=PT_SMALL)
    ax.set_xticks([0,.5,1],["0","0.5","1"])
    ax.set_xlabel("Task × credit contrast (NMSE)",fontsize=PT_LABEL)
    ax.text(.98,.99,"Selected = common rates",ha="right",va="top",transform=ax.transAxes,fontsize=PT_SMALL)


def cancellation_panel(ax,diagnostics,table):
    chosen=diagnostics[diagnostics.task.eq("opposed_strong")&diagnostics.source_rule.eq("calibrated_broadcast")&diagnostics.parameter_scope.eq("distal_parameters")]
    for rule,label,marker,offset in [("exact","Exact path","o",-.035),("calibrated_broadcast","Calibrated broadcast","s",.035)]:
        rows=[]
        for i,state in enumerate(["initial","extended_best"]):
            part=chosen[chosen.state.eq(state)&chosen.evaluated_rule.eq(rule)].sort_values("seed")
            assert len(part)==20 and set(part.seed)==EXPECTED_SEEDS
            result=interval(part.context_cancellation_ratio)
            rows.append(result)
            table.append(dict(panel="F",task="opposed_strong",state=state,source_rule="calibrated_broadcast",evaluated_rule=rule,metric="context_cancellation_ratio",**result))
        mean=np.array([row["mean"] for row in rows]);lo=np.array([row["ci_low"] for row in rows]);hi=np.array([row["ci_high"] for row in rows])
        ax.errorbar(np.array([0,1])+offset,mean,yerr=[mean-lo,hi-mean],color=COLOR[rule],marker=marker,
                    lw=LW_DATA,ms=MARKER_MS,capsize=ERR_CAPSIZE,label=label)
    ax.set_xlim(-.18,1.18);ax.set_ylim(0,1.03)
    ax.set_xticks([0,1],["Initial","After training"])
    ax.set_yticks([0,.5,1],["0","0.5","1"])
    ax.set_ylabel("Gradient retained after\ncontext averaging",fontsize=PT_LABEL,labelpad=1)
    ax.legend(loc="upper right",fontsize=PT_SMALL,frameon=False,handlelength=1.8,borderaxespad=.2)


def build():
    DEST.mkdir(parents=True,exist_ok=True)
    audit=json.loads((SOURCE/"audit.json").read_text())
    assert audit["all_outcome_hashes_pass"] and audit["outcome_counts"]==dict(development=288,fresh=320,extension=320)
    curves=pd.read_csv(SOURCE/"all_curves.csv")
    curves=curves[curves.phase.isin(["fresh","extension"])&curves.optimizer.eq("adam")&curves.selected_rate].copy()
    # The continuation records the shared 4,096 checkpoint again. Verify the
    # duplicate outcomes before retaining one row per paired seed/condition.
    keys=["seed","task","optimizer","rule","rate","step"]
    duplicated=curves[curves.duplicated(keys,keep=False)]
    assert (duplicated.groupby(keys).test_nmse.agg(lambda x: float(x.max()-x.min()))<1e-13).all()
    curves=curves.drop_duplicates(keys,keep="first")
    contrasts=pd.read_csv(SOURCE/"paired_contrasts.csv")
    seeds=pd.read_csv(SOURCE/"paired_seed_contrasts.csv")
    gradients=pd.read_csv(SOURCE/"context_gradient_summary.csv")
    gradient_audit=json.loads((SOURCE/"context_gradient_audit.json").read_text())
    assert gradient_audit["maximum_absolute_error"]<1e-10
    table=[]
    canvas=NativeCanvas(490/72,3,row_weights=[112,122,113],hgutter_pt=37,vgutter_pt=46,
                        margins=Margins(left=62,right=15,top=27,bottom=40))
    a=canvas.panel("A",0,0,6,schematic=True,title="Seven-compartment conductance tree")
    b=canvas.panel("B",0,6,6,schematic=True,title="Teacher tuning defines the task")
    c=canvas.panel("C",1,0,6,title="Aligned branch tuning",grid="y")
    d=canvas.panel("D",1,6,6,title="Opposed branch tuning",grid="y")
    e=canvas.panel("E",2,0,6,title="The credit deficit survives extension",grid="x")
    f=canvas.panel("F",2,6,6,title="Broadcast gradients cancel",grid="y")
    circuit(a);schematic_rows=tuning(b)
    overlaps={task:curves_panel(ax,task,curves,table,legend=task==TASKS[0]) for task,ax in zip(TASKS,[c,d])}
    contrasts_panel(e,contrasts,seeds,table)
    cancellation_panel(f,gradients,table)
    style_direct_color_labels(canvas.fig)
    output=DEST/"conductance_opponent_credit_native.pdf"
    result=canvas.save(output,name="conductance_opponent_credit_native",dpi=200)
    plt.close(canvas.fig)
    pd.DataFrame(table).to_csv(DEST/"figure_source.csv",index=False)
    pd.DataFrame(schematic_rows).to_csv(DEST/"nominal_teacher_tuning.csv",index=False)
    sources=[SOURCE/name for name in ["all_curves.csv","paired_contrasts.csv","paired_seed_contrasts.csv","context_gradient_summary.csv","context_gradient_audit.json","audit.json"]]
    sources += [ROOT/"protocol.json",ROOT/"selection_freeze.json",HERE/"opponent_model.py"]
    provenance=dict(figure_sha256=sha(output),builder_sha256=sha(Path(__file__)),
        source_sha256={str(path.relative_to(JOURNAL)):sha(path) for path in sources},panels="A–F",n_seed_blocks=20,
        native_width_pt=518.4,native_height_pt=490,
        learning_curves="All 20 selected-rate Adam fresh fits plus exact optimizer-state continuations; repeated shared checkpoint retained once after numerical agreement check. Curves show fixed-checkpoint test NMSE, not validation-selected endpoints.",
        contrasts="Calibrated-broadcast minus exact deficit on opposed minus aligned targets, paired within seed. Validation-selected saved states within each budget. Selected and common rate rows are numerically identical and displayed once.",
        diagnostic="Opposed task, distal 16 log-conductance parameters, all evaluated rules at the same calibrated-broadcast parameter state. Retained fraction = norm(sum_c p_c g_c) / sum_c p_c norm(g_c).",
        schematic="B shows nominal unjittered teacher leaf voltage E/(1+E+I), E coefficients (8,.25) and I coefficients (.25,8), exchanged in the right subtree for opposed tuning. Actual seed-specific teachers add the frozen log-conductance jitter.",
        broadcast_maximum_paired_curve_absolute_difference=overlaps,
        interpretation="Three fixed initial-profile patterns with oracle coefficients; binary-context current path rank at most two. No six-independent-errors, reciprocal-cable or quartic-representability claim.",
        confidence_intervals="Descriptive 95% percentile intervals from 20000 paired whole-seed resamples, seed 982211; no new hypothesis test.")
    (DEST/"figure_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    return result


if __name__=="__main__":
    build()
