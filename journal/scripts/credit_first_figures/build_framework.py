#!/usr/bin/env python3
"""Credit-first Fig. 1 with the frozen six-arm, fresh-seed MNIST ladder.

Panel A reuses the public eligibility drawing. Panel B depicts the actual
12-site projection dictionaries. C--D read the complete six-arm follow-up;
E replays separate legacy random-feedback/CIFAR paired contrasts. This
builder performs no fitting, rate selection, or confidence-interval fitting.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
import build_main_figure_01 as framework
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF,
                           MARKER_MS, PT_ANNOT, PT_LABEL, PT_LEGEND, PT_SMALL,
                           Margins, NativeCanvas, style_panel)
from journal_style import style_direct_color_labels
from native_schematics import Frame

SOURCE = JOURNAL / "source_data"
FRESH = SOURCE / "image_ladder_controls/summaries"
OUT = JOURNAL / "figures/components/credit_first_figure_01.pdf"
RECORDS = SOURCE / "credit_first_figures"
ARMS = ("strict_scalar", "neuron_shared", "projected_k1", "subtree_k3", "exact_path", "decoder_only")
ARM_LABELS = ("Layer\nscalar", "Per\nneuron", "Projected\nK = 1", "Subtrees\nK = 3", "Exact\npath", "Decoder\nonly")
ARCHITECTURES = ("shunting", "additive")
CONTRASTS = ("subtree_k3_minus_projected_k1", "exact_path_minus_subtree_k3")


def dictionaries(ax):
    """Actual three-proximal/nine-distal morphology and projection matrices.

    Native row indices follow the model: 0--2 proximal, 3--11 distal. For
    readability the matrices are displayed in subtree order, with each
    proximal site immediately before its three distal children.
    """
    f = Frame(ax)
    root = (.027, .52)
    subtree_colors = [COLORS["additive"], COLORS["shunting"], COLORS["bp"]]
    ax.text(.53, .96, "Delivered field = A c", ha="center", fontsize=PT_LABEL)
    for k, yc in enumerate((.74, .51, .28)):
        prox = (.12, yc)
        color = subtree_colors[k]
        ax.plot([root[0], prox[0]], [root[1], prox[1]], color=color, lw=LW_DATA)
        f.disc(prox, 2.1, fill=color)
        for j in range(3):
            leaf = (.255, yc + (1-j)*.064)
            ax.plot([prox[0], leaf[0]], [prox[1], leaf[1]], color=color, lw=LW_EDGE)
            f.disc(leaf, 1.65, fill=color)
    f.disc(root, 3.0, fill=COLORS["panel_bg"], edge=COLORS["edge"], lw=LW_EDGE)
    ax.text(.13, .135, "12 sites", ha="center", fontsize=PT_ANNOT)
    ax.text(.13, .058, "+ soma", ha="center", fontsize=PT_SMALL, color=COLORS["mute"])
    a1 = np.ones((12, 1))
    a3 = np.zeros((12, 3))
    for k in range(3):
        a3[[k, 3+3*k, 4+3*k, 5+3*k], k] = 1
    a12 = np.eye(12)
    order = np.array([0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11])
    cmap = LinearSegmentedColormap.from_list("delivery", ["white", COLORS["shunting"]])
    boxes = [(.355, .065), (.515, .115), (.73, .24)]
    for a, (x, width), label in zip((a1, a3, a12), boxes, ("K = 1", "K = 3", "K = 12")):
        data = a[order]
        if a.shape[1] == 12:
            data = data[:, order]
        inner = ax.inset_axes([x, .21, width, .59])
        if a.shape[1] == 3:
            rgba = np.ones((*data.shape, 4))
            for k, color in enumerate(subtree_colors):
                rgba[data[:, k].astype(bool), k] = to_rgba(color)
            inner.imshow(rgba, aspect="auto", interpolation="nearest")
        else:
            inner.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        inner.set_xticks([]); inner.set_yticks([])
        for y in (3.5, 7.5):
            inner.axhline(y, color=COLORS["edge"], lw=LW_HAIR)
        for spine in inner.spines.values():
            spine.set_color(COLORS["edge"]); spine.set_linewidth(LW_HAIR)
        ax.text(x+width/2, .12, label, ha="center", fontsize=PT_ANNOT)
    ax.text(.665, .03, "K = 1 and 3: oracle projection", ha="center", fontsize=PT_SMALL)
    ax.text(.53, -.065, "Rows: sites grouped by subtree; columns: profiles", ha="center", fontsize=PT_SMALL, color=COLORS["mute"])
    return dict(broadcast=a1, subtrees=a3, resolved=a12, display_row_order=order)


def read_fresh():
    conditions = pd.read_csv(FRESH / "condition_summary_six_rules.csv", float_precision="round_trip")
    seeds = pd.read_csv(FRESH / "fresh_analysis_rows_six_rules.csv", float_precision="round_trip")
    paired = pd.read_csv(FRESH / "paired_contrasts_six_rules.csv", float_precision="round_trip")
    conditions = conditions[conditions.metric.eq("test_accuracy") & conditions.rate_policy.eq("selected")].copy()
    seeds = seeds[seeds.rate_policy.eq("selected")].copy()
    paired = paired[paired.metric.eq("test_accuracy") & paired.rate_policy.eq("selected") & paired.contrast.isin(CONTRASTS)].copy()
    assert len(seeds) == 120 and len(conditions) == 12 and len(paired) == 4
    assert seeds.epochs.eq(180).all()
    assert seeds.groupby(["architecture", "seed"]).initialized_model_sha256.nunique().eq(1).all()
    assert seeds[seeds.arm.eq("decoder_only")].decoder_only_core_unchanged.all()
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            s = seeds[seeds.architecture.eq(architecture) & seeds.arm.eq(arm)]
            row = conditions[conditions.architecture.eq(architecture) & conditions.arm.eq(arm)]
            assert len(s) == 10 and s.seed.nunique() == 10, (architecture, arm, len(s))
            assert len(row) == 1 and int(row.iloc[0].n) == 10
            assert abs(s.test_accuracy.mean() - row.iloc[0]["mean"]) < 1e-12
        p = seeds[seeds.architecture.eq(architecture)].pivot(index="seed", columns="arm", values="test_accuracy")
        assert p.shape == (10, 6) and not p.isna().any().any()
        for name in CONTRASTS:
            lhs, rhs = name.split("_minus_")
            row = paired[paired.architecture.eq(architecture) & paired.contrast.eq(name)]
            assert len(row) == 1 and int(row.iloc[0].n) == 10
            assert abs((p[lhs]-p[rhs]).mean() - row.iloc[0]["mean"]) < 1e-12
    return conditions, seeds, paired


def accuracy(ax, conditions, seeds):
    positions = np.array([0, 1, 2, 3, 4, 5.7])
    ax.axvspan(5.13, 6.15, color=COLORS["panel_bg"], zorder=0)
    for architecture, offset, marker in [("shunting", -.09, "o"), ("additive", .09, "s")]:
        color = COLORS[architecture]
        group = seeds[seeds.architecture.eq(architecture)]
        pivot = group.pivot(index="seed", columns="arm", values="test_accuracy").loc[:, list(ARMS)]
        for _, seed in pivot.iterrows():
            ax.plot(positions[:5]+offset, 100*seed.iloc[:5], color=color, alpha=.16, lw=LW_HAIR, zorder=1)
        for x, arm in zip(positions, ARMS):
            values = 100*group[group.arm.eq(arm)].sort_values("seed").test_accuracy.to_numpy()
            ax.scatter(x+offset+np.linspace(-.038,.038,10), values, color=color, s=5.7, alpha=.35, zorder=2, linewidths=0)
            row = conditions[conditions.architecture.eq(architecture) & conditions.arm.eq(arm)].iloc[0]
            ax.errorbar(x+offset, 100*row["mean"], yerr=[[100*(row["mean"]-row.ci_low)], [100*(row.ci_high-row["mean"])]],
                        fmt=marker, color=color, ms=MARKER_MS, mfc="white", mew=LW_ERR,
                        elinewidth=LW_ERR, capsize=2, zorder=4)
    floor = max(0, 5*np.floor((100*seeds.test_accuracy.min()-7)/5))
    ax.set(xlim=(-.43,6.13), ylim=(floor,100.5), xticks=positions, xticklabels=ARM_LABELS, ylabel="Test accuracy (%)")
    style_panel(ax, grid="y")
    handles = [Line2D([],[],color=COLORS[a],marker=m,mfc="white",lw=LW_DATA,ms=MARKER_MS,label=l)
               for a,m,l in [("shunting","o","Shunting"),("additive","s","Raw additive")]]
    ax.legend(handles=handles,loc="lower left",bbox_to_anchor=(.025,.035),frameon=False,fontsize=PT_LEGEND,ncol=2,columnspacing=1.5,handlelength=1.7)


def resolution(ax, paired, seeds):
    ax.axhline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    for architecture,offset,marker in [("shunting",-.13,"o"),("additive",.13,"s")]:
        color = COLORS[architecture]
        pivot = seeds[seeds.architecture.eq(architecture)].pivot(index="seed",columns="arm",values="test_accuracy")
        for x,name in enumerate(CONTRASTS):
            lhs,rhs = name.split("_minus_")
            values = 100*(pivot[lhs]-pivot[rhs])
            ax.scatter(x+offset+np.linspace(-.045,.045,len(values)),values,s=5.7,color=color,alpha=.30,linewidths=0,zorder=1)
            r = paired[paired.architecture.eq(architecture)&paired.contrast.eq(name)].iloc[0]
            ax.errorbar(x+offset,100*r["mean"],yerr=[[100*(r["mean"]-r.ci_low)],[100*(r.ci_high-r["mean"])]],
                        fmt=marker,color=color,mfc="white",mew=LW_ERR,ms=MARKER_MS,elinewidth=LW_ERR,capsize=2,zorder=3)
    ax.set(xlim=(-.45,1.45),xticks=[0,1],xticklabels=["K = 3 − K = 1","Exact − K = 3"],ylabel="Paired difference (pp)")
    style_panel(ax,grid="y")


def legacy_contrasts():
    factorial = pd.read_csv(SOURCE / "mnist_between_within_factorial/paired_contrasts.csv")
    cifar = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv")
    rows=[]
    for architecture in ARCHITECTURES:
        key="dfa within: exact path - neuron"
        r=factorial[factorial.architecture.eq(architecture)&factorial.contrast.eq(key)].iloc[0]
        rows.append(dict(task="MNIST",source="DFA",architecture=architecture,
                         label=f"DFA: {'shunt' if architecture=='shunting' else 'additive'}",
                         mean_pp=100*r.mean_difference,low_pp=100*r.ci95_low,high_pp=100*r.ci95_high,n_seeds=int(r.n_seeds),
                         source_table="source_data/mnist_between_within_factorial/paired_contrasts.csv",source_contrast=key))
    key="exact path minus neuron specific"
    r=cifar[cifar.contrast.eq(key)].iloc[0]
    rows.append(dict(task="CIFAR-10",source="Readout",architecture="additive",label="CIFAR: additive",
                     mean_pp=100*r.mean_difference,low_pp=100*r.ci95_low_difference,high_pp=100*r.ci95_high_difference,n_seeds=int(r.n_seeds),
                     source_table="source_data/cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv",source_contrast=key))
    return pd.DataFrame(rows)


def legacy_forest(ax,rows):
    ax.axvline(0,color=COLORS["mute"],lw=LW_REF,ls="--",zorder=0)
    for y,r in zip([2,1,-.3],rows.itertuples()):
        color=COLORS[r.architecture]
        ax.errorbar(r.mean_pp,y,xerr=[[r.mean_pp-r.low_pp],[r.high_pp-r.mean_pp]],
                    fmt="o" if r.architecture=="shunting" else "s",color=color,ms=MARKER_MS,
                    elinewidth=LW_ERR,capsize=2,mfc="white",mew=LW_ERR)
    ax.set(yticks=[2,1,-.3],yticklabels=rows.label,ylim=(-.85,2.55),xlim=(-1.32,.39),xticks=[-1,-.5,0],xlabel="Exact − neuron-specific (pp)")
    ax.axhline(.35,color=COLORS["grid"],lw=LW_HAIR)
    style_panel(ax,grid="x")
    ax.tick_params(axis="y",length=0,labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)


def main():
    conditions,seeds,paired=read_fresh()
    RECORDS.mkdir(exist_ok=True)
    canvas=NativeCanvas(490/72,3,row_weights=[126,133,104],hgutter_pt=40,vgutter_pt=42,
                        margins=Margins(left=36,right=12,top=24,bottom=39))
    a=canvas.panel("A",0,0,6,schematic=True,title="Eligibility × delivered credit",lock=False)
    b=canvas.panel("B",0,6,6,schematic=True,title="One tree, three dictionaries",lock=False)
    c=canvas.panel("C",1,0,12,title="MNIST: ten paired seeds at independently selected rates")
    d=canvas.panel("D",2,0,6,title="Additional within-tree resolution")
    e=canvas.panel("E",2,6,6,title="Separate image controls")
    framework.panel_factorization_readable(a)
    a.set_xlim(0, .88)  # Use the half-width schematic cell without shrinking type.
    matrices=dictionaries(b)
    accuracy(c,conditions,seeds)
    resolution(d,paired,seeds)
    legacy=legacy_contrasts();legacy_forest(e,legacy)
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()
    # The two lower axes span six grid modules and therefore share one width.
    lower_left = max(canvas._locks[name][0] for name in ("D", "E"))
    lower_right = max(canvas._locks[name][1] for name in ("D", "E"))
    for panel in (d, e):
        canvas.declare_reserve(panel, left=lower_left, right=lower_right)
    canvas.lock_reserves()
    right_letters=[item["art"] for item in canvas._letters if item["letter"] in "BE"]
    shared_x=min(art.get_position()[0] for art in right_letters)
    for art in right_letters:art.set_position((shared_x,art.get_position()[1]))
    problems=canvas.save(OUT,name="credit_first_figure_01",dpi=180,lock=False)
    np.savez_compressed(RECORDS/"figure_01_illustrative_dictionaries.npz",**matrices)
    legacy.to_csv(RECORDS/"figure_01_contrasts.csv",index=False)
    plotted=[]
    for panel,table,rows in [("C","condition_summary_six_rules.csv",conditions),
                            ("D","paired_contrasts_six_rules.csv",paired)]:
        plotted.extend(dict(panel=panel,source_table=str((FRESH/table).relative_to(JOURNAL)),**r)for r in rows.to_dict("records"))
    plotted.extend(dict(panel="E",**r)for r in legacy.to_dict("records"))
    pd.DataFrame(plotted).to_csv(RECORDS/"figure_01_six_arm_source.csv",index=False)
    seeds.to_csv(RECORDS/"figure_01_six_arm_seed_source.csv",index=False)
    files=[Path(__file__),Path(framework.__file__),JOURNAL/"scripts/figure_canvas.py",JOURNAL/"scripts/journal_style.py",
           *[FRESH/name for name in ["condition_summary_six_rules.csv","fresh_analysis_rows_six_rules.csv","paired_contrasts_six_rules.csv"]],
           SOURCE/"mnist_between_within_factorial/paired_contrasts.csv",SOURCE/"cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv",
           *[SOURCE/"image_ladder_controls"/name for name in ["protocol.json", "selection.json", "projected_k1/protocol.json", "projected_k1/selection.json"]]]
    payload=dict(panel_sources={
        "A":"Public build_main_figure_01.panel_factorization_readable: directed-tree eligibility and exact path example.",
        "B":"Actual 12 nonsomatic sites: three proximal sites and nine distal children; soma outside basis. K1/K3 oracle projection in activation coordinates; matrices displayed in subtree order. K12 resolved dictionary spans arbitrary fields and does not imply 12 independently supplied external errors.",
        "C":"Complete six-arm fresh MNIST selected-rate cohort: 10 paired seeds per architecture; 180 epochs; validation-selected checkpoint. Decoder-only core remains fixed and is separated from credit conditions.",
        "D":"Fresh paired test-accuracy contrasts K3 minus projected K1 and exact path minus K3, same selected-rate cohort as C.",
        "E":"Separate legacy MNIST direct-feedback-alignment and flattened CIFAR-10 paired exact-minus-neuron contrasts; source cohorts kept separate from C--D."},
        source_sha256={str(p.relative_to(JOURNAL)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files},
        numerical_scope="No fitting, selection, bootstrap, or source-outcome mutation. Existing 95% bootstrap intervals replayed exactly; seed means and paired contrast means independently recomputed as assertions.",
        plot_units={"C":"Test accuracy (%); source fractions multiplied by 100",
                    "D":"Paired test-accuracy differences (percentage points); source fractions multiplied by 100",
                    "E":"Legacy paired test-accuracy differences (percentage points)"},
        fresh_seed_count_per_architecture=10,normal_font_minimum_pt=PT_SMALL,layout_findings=problems)
    (RECORDS/"figure_01_sources.json").write_text(json.dumps(payload,indent=2)+"\n")
    caption=r"""\textbf{Task-derived credit separates neuronal identity from resolution within a dendritic tree.}
\textbf{A}, Conductance updates combine local eligibility $e_i$ and delivered compartment credit $\varepsilon_n$. Here $\eta$ is learning rate, $x_i$ presynaptic activity, $R_n$ input resistance, and $E_i-V_n$ driving force. Exact directed-tree transport multiplies somatic error $\delta_0$ by path derivatives $\alpha$; reciprocal cables require an adjoint solution.
\textbf{B}, Over the twelve nonsomatic sites, $K=1$ broadcasts, $K=3$ groups each proximal site with its three children, and $K=12$ resolves every site. Projected rules retain exact somatic errors. The $K=1$ and $K=3$ training conditions use oracle projection of the exact activation-error field. $K$ counts spatial profiles, not independently computed external errors.
\textbf{C}, MNIST with one 128-neuron dendritic layer and a linear readout. Neuron-specific credit broadcasts each neuron's exact somatic activation error; the strict scalar removes neuronal identity. The decoder-only reference freezes dendritic parameters. Thin lines pair ten fresh seeds per architecture; large symbols show means and 95\% descriptive intervals. All fits use 180 epochs, rates selected on three separate development seeds, and validation-selected checkpoints.
\textbf{D}, Paired accuracy differences compare $K=3$ with projected $K=1$, and exact paths with $K=3$. The projected rules use the same exact-field coefficient source and preserve the same mean at identical states, but their norms can differ. Points show seeds with paired 95\% intervals.
\textbf{E}, Separate legacy cohorts compare exact paths with neuron-specific broadcast under fixed random between-neuron feedback on MNIST (direct feedback alignment, DFA; 15 seeds) or a readout derivative on flattened CIFAR-10 (20 seeds). Colors and shapes follow \textbf{C}. Percentage points are abbreviated pp; original 15-seed readout-derived MNIST ladders remain in Supplementary Fig.~S45. Supplementary Fig.~S49 adds common-rate and capture controls.
"""
    (RECORDS/"figure_01_caption.tex").write_text(caption)
    (RECORDS/"figure_01_caption.md").write_text(caption)
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
