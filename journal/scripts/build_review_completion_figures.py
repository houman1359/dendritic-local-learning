#!/usr/bin/env python3
"""Native figures for new coefficient and morphology-uncertainty analyses."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import journal_style as J
from journal_style import (apply_neurips_style, LW_DATA, LW_REF, LW_EDGE,
                           LW_HAIR, PT_BASE, PT_EMPH, PT_LETTER)
from figure_canvas import enforce_tokens
from neurips_style import FIG_W

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures/supplementary"
# Condition inks.  Every chromatic ink is a NAMED entry of a journal register,
# so the strict audit's role check sees a registered role rather than a hue
# invented here that sits a few OKLab units from one (the 2026-09-08 audit read
# #16817a as series:ordinal3, #a57422 as series:low_rank and #9b4b6d as
# anatomy:inh).  Each condition keeps the hue family, and so the meaning, it
# already had: the learned soft rule stays the teal that no rule series claims
# (ORDINAL_RAMP's dark step, ΔE 6.5 from the old teal), the frozen-profile
# reference stays ochre (low_rank, ΔE 7.9), the mismatched control stays the
# muted plum-carmine (the inhibitory/gate carmine, ΔE 10.3).  The hard-readout
# blue is a sample of the journal sequential ramp itself (ΔE 0.66 from
# SEQ_CMAP) and the oracle line is the achromatic reference ink, so neither is
# a competing hue.
COLORS={"oracle_context":"#333333","learned_local_cue":J.ORDINAL_RAMP[3],
        "frozen_profile":J.SERIES_COLORS["low_rank"],
        "mismatched_encoder":J.ANATOMY_COLORS["inh"],"hard":"#376cb0"}
LABELS={"oracle_context":"Oracle context","learned_local_cue":"Learned, soft",
        "frozen_profile":"Frozen profile","mismatched_encoder":"Mismatched"}


def style():
    apply_neurips_style()
    # Journal face and the three-token type scale (7 / 8 / 9 pt).
    plt.rcParams.update({"font.size":PT_EMPH,"axes.titlesize":PT_EMPH,
        "axes.labelsize":PT_EMPH,"xtick.labelsize":PT_BASE,"ytick.labelsize":PT_BASE,
        "legend.fontsize":PT_BASE,
        # Stroke tokens: the journal base style still ships the 0.8 pt spine and
        # tick of the frozen NeurIPS layer, which is not a line-weight token.
        "axes.linewidth":LW_EDGE,"xtick.major.width":LW_EDGE,
        "ytick.major.width":LW_EDGE,"xtick.minor.width":LW_HAIR,
        "ytick.minor.width":LW_HAIR,"patch.linewidth":LW_HAIR,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})


def panel(ax,letter,title,x=-.16):
    ax.set_title(title,loc="left",pad=10)
    ax.text(x,1.09,letter,transform=ax.transAxes,fontweight="bold",fontsize=PT_LETTER)


def interval(v):
    v=np.asarray(v,float);rng=np.random.default_rng(83710)
    draw=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)
    return v.mean(),*np.quantile(draw,[.025,.975])


def curve(ax,frame,x,metric,color,label,marker="o",ls="-"):
    values=[]
    for at,g in frame.groupby(x):values.append((at,*interval(g[metric])))
    values=np.asarray(values)
    ax.errorbar(values[:,0],values[:,1],yerr=np.stack([values[:,1]-values[:,2],values[:,3]-values[:,1]]),
                color=color,label=label,marker=marker,ms=3.5,lw=LW_DATA,capsize=2,ls=ls)


SEED_JITTER=np.random.default_rng(4471)
MARKER={"learned_local_cue":"o","mismatched_encoder":"o","hard":"s"}
REFERENCE={"oracle_context":"-.","frozen_profile":":"}


def seeded_curve(ax,frame,x,metric,color,label,marker="o",ls="-",positions=None,dodge=0.,jitter=.03):
    """Mean curve with 95% paired-seed bootstrap bars and every seed drawn behind it.

    ``positions`` maps the factor levels to ordinal x positions (a handful of
    levels is drawn at equal spacing); ``dodge`` separates series whose means
    coincide at the marker scale.
    """
    xpos=(lambda v:positions[v]) if positions else (lambda v:v)
    values=[]
    for at,g in frame.groupby(x):
        seeds=g[metric].to_numpy(float)
        ax.scatter(xpos(at)+dodge+SEED_JITTER.normal(0,jitter,len(seeds)),seeds,s=7,marker=marker,
                   color=color,alpha=.3,edgecolors="none",zorder=2)
        values.append((xpos(at),*interval(seeds)))
    values=np.asarray(values)
    ax.errorbar(values[:,0]+dodge,values[:,1],yerr=np.stack([values[:,1]-values[:,2],values[:,3]-values[:,1]]),
                color=color,label=label,marker=marker,ms=3.5,lw=LW_DATA,capsize=2,ls=ls,
                markeredgecolor="white",markeredgewidth=LW_HAIR,zorder=4)


def reference_line(ax,frame,metric,color,label,ls,note_x,va="bottom"):
    """A condition-invariant control drawn as one labelled reference line.

    The mean is required to be identical at every level of the panel factor
    (it is, per seed, by construction); the band is its 95% seed interval.
    """
    per_level=frame.groupby(["calibration_samples","cue_noise_sd","cue_delay_trials"])[metric].mean()
    if not np.allclose(per_level.to_numpy(),per_level.iloc[0]):
        raise ValueError(f"{label} varies across the panel factor; draw it as a series")
    seeds=frame.groupby("seed")[metric].mean().to_numpy(float)
    m,lo,hi=interval(seeds)
    ax.axhspan(lo,hi,color=color,alpha=.12,lw=0,zorder=1)
    ax.axhline(m,color=color,ls=ls,lw=LW_REF,zorder=3,label=label)
    ax.text(note_x,m+(1.2 if va=="bottom" else -1.2),f"{label} {m:.1f}%",ha="center",va=va,
            fontsize=PT_BASE,color=color,zorder=5)


def encoder():
    soft=pd.read_csv(ROOT/"source_data/review_coefficient_encoder/trajectories.csv")
    hard=pd.read_csv(ROOT/"source_data/review_coefficient_hard_readout/trajectories.csv")
    for f in [soft,hard]:f["accuracy_percent"]=100*f.heldout_accuracy
    final=soft[soft.epoch.eq(80)];hfinal=hard[hard.epoch.eq(80)]
    fig,axs=plt.subplots(3,2,figsize=(FIG_W,7.0))
    fig.subplots_adjust(left=.105,right=.975,bottom=.08,top=.875,wspace=.34,hspace=.67)
    ax=axs[0,0];ax.axis("off");panel(ax,"A","Learning a route coefficient")
    boxes=[(.01,.66,.97,.23,"Local branch cue + noise\nSeparate calibration activation targets"),
           (.01,.26,.97,.23,"Four-output context estimator\nFrozen before task learning")]
    for x,y,w,h,text in boxes:
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",transform=ax.transAxes,fontsize=PT_EMPH,
                bbox={"boxstyle":"round,pad=.5","fc":"#f0f5f4","ec":"#a1b5b2","lw":LW_HAIR})
    ax.add_patch(FancyArrowPatch((.5,.64),(.5,.51),transform=ax.transAxes,arrowstyle="->",mutation_scale=11,color="#405a55",lw=LW_EDGE))
    ax.text(.5,-.04,"Soft coefficients → four subtree profiles\nMultiply by scalar error and local eligibility",ha="center",va="center",transform=ax.transAxes,fontsize=PT_EMPH)
    ax=axs[0,1];panel(ax,"B","Cue noise limits learning")
    f=final[final.calibration_samples.eq(256)&final.cue_delay_trials.eq(0)]
    for method in LABELS:curve(ax,f[f.method.eq(method)],"cue_noise_sd","accuracy_percent",COLORS[method],LABELS[method],ls=REFERENCE.get(method,"-"))
    h=hfinal[hfinal.calibration_samples.eq(256)&hfinal.cue_delay_trials.eq(0)&hfinal.method.eq("learned_local_cue")]
    curve(ax,h,"cue_noise_sd","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--")
    ax.set(xlabel="Cue noise SD",ylabel="Held-out accuracy (%)",ylim=(10,90))
    # Shared key with the real plot glyphs of panels C and D (reference lines
    # for the two condition-invariant controls, square + dashed for the
    # exploratory hard readout), plus the asterisk footnote beneath it.
    key=[Line2D([],[],color=COLORS["oracle_context"],ls=REFERENCE["oracle_context"],lw=LW_REF,label="Oracle context"),
         Line2D([],[],color=COLORS["learned_local_cue"],marker="o",ms=3.5,lw=LW_DATA,label="Learned, soft"),
         Line2D([],[],color=COLORS["frozen_profile"],ls=REFERENCE["frozen_profile"],lw=LW_REF,label="Frozen profile"),
         Line2D([],[],color=COLORS["mismatched_encoder"],marker="o",ms=3.5,lw=LW_DATA,label="Mismatched"),
         Line2D([],[],color=COLORS["hard"],marker="s",ms=3.5,lw=LW_DATA,ls="--",label="Learned, hard*")]
    fig.legend(handles=key,frameon=False,loc="upper center",bbox_to_anchor=(.54,.995),ncol=3,columnspacing=1.5,handlelength=1.8)
    fig.text(.54,.947,"* maximum-probability route selection by the same frozen encoder (exploratory paired sensitivity)",
             ha="center",va="top",fontsize=PT_BASE,color="#555555")
    # Row 2 is the pair the consolidated supplement reproduces: equal axes
    # widths, one shared accuracy axis, every seed drawn, ordinal x axes.
    row_y=axs[1,0].get_position().y0;row_h=axs[1,0].get_position().height
    axs[1,0].set_position([42/518.4,row_y,220/518.4,row_h]);axs[1,1].set_position([282/518.4,row_y,220/518.4,row_h])
    ax=axs[1,0];panel(ax,"C","Calibration data and computation")
    f=final[final.cue_noise_sd.eq(.5)&final.cue_delay_trials.eq(0)]
    positions={16:0,64:1,256:2}
    reference_line(ax,f[f.method.eq("oracle_context")],"accuracy_percent",COLORS["oracle_context"],"oracle",REFERENCE["oracle_context"],0.5,va="bottom")
    reference_line(ax,f[f.method.eq("frozen_profile")],"accuracy_percent",COLORS["frozen_profile"],"frozen",REFERENCE["frozen_profile"],0.5,va="top")
    seeded_curve(ax,f[f.method.eq("mismatched_encoder")],"calibration_samples","accuracy_percent",COLORS["mismatched_encoder"],LABELS["mismatched_encoder"],positions=positions,dodge=.07)
    seeded_curve(ax,f[f.method.eq("learned_local_cue")],"calibration_samples","accuracy_percent",COLORS["learned_local_cue"],LABELS["learned_local_cue"],positions=positions,dodge=-.07)
    h=hfinal[hfinal.cue_noise_sd.eq(.5)&hfinal.cue_delay_trials.eq(0)&hfinal.method.eq("learned_local_cue")]
    seeded_curve(ax,h,"calibration_samples","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--",positions=positions)
    ax.set(xlabel="Calibration examples (30 epochs)",ylabel="Held-out accuracy (%)",ylim=(13,87),xlim=(-.45,2.45))
    ax.set_xticks([0,1,2],["16","64","256"]);ax.set_yticks([20,40,60,80])
    # D shares C's accuracy axis, so its letter sits just left of its spine.
    ax=axs[1,1];panel(ax,"D","Cue / eligibility timing mismatch",x=-.055)
    f=final[final.calibration_samples.eq(256)&final.cue_noise_sd.eq(.5)]
    positions={0:0,1:1,4:2}
    reference_line(ax,f[f.method.eq("oracle_context")],"accuracy_percent",COLORS["oracle_context"],"oracle",REFERENCE["oracle_context"],0.5,va="bottom")
    reference_line(ax,f[f.method.eq("frozen_profile")],"accuracy_percent",COLORS["frozen_profile"],"frozen",REFERENCE["frozen_profile"],0.5,va="top")
    seeded_curve(ax,f[f.method.eq("mismatched_encoder")],"cue_delay_trials","accuracy_percent",COLORS["mismatched_encoder"],LABELS["mismatched_encoder"],positions=positions,dodge=.07)
    seeded_curve(ax,f[f.method.eq("learned_local_cue")],"cue_delay_trials","accuracy_percent",COLORS["learned_local_cue"],LABELS["learned_local_cue"],positions=positions,dodge=-.07)
    h=hfinal[hfinal.calibration_samples.eq(256)&hfinal.cue_noise_sd.eq(.5)&hfinal.method.eq("learned_local_cue")]
    seeded_curve(ax,h,"cue_delay_trials","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--",positions=positions)
    ax.text(1.5,50,"soft, hard and mismatched\ncoincide at delays 1 and 4\n(within 1 pp)",ha="center",va="center",fontsize=PT_BASE,color="#555555",style="italic")
    ax.set(xlabel="Cue delay (independent trials)",ylim=(13,87),xlim=(-.45,2.45))
    ax.set_xticks([0,1,2],["0","1","4"]);ax.set_yticks([20,40,60,80]);ax.tick_params(axis="y",labelleft=False)
    ax=axs[2,0];panel(ax,"E","Task-learning trajectories")
    f=soft[soft.calibration_samples.eq(256)&soft.cue_noise_sd.eq(.5)&soft.cue_delay_trials.eq(0)]
    for method in LABELS:curve(ax,f[f.method.eq(method)],"epoch","accuracy_percent",COLORS[method],LABELS[method],ls=REFERENCE.get(method,"-"))
    h=hard[hard.calibration_samples.eq(256)&hard.cue_noise_sd.eq(.5)&hard.cue_delay_trials.eq(0)&hard.method.eq("learned_local_cue")]
    curve(ax,h,"epoch","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--")
    ax.set(xlabel="Task epoch",ylabel="Held-out accuracy (%)",ylim=(10,90))
    ax=axs[2,1];panel(ax,"F","Primary soft-estimator contrasts")
    f=final[final.calibration_samples.eq(256)&final.cue_noise_sd.eq(.5)&final.cue_delay_trials.eq(0)]
    wide=f.pivot(index="seed",columns="method",values="accuracy_percent")
    for i,method in enumerate(["oracle_context","frozen_profile","mismatched_encoder"]):
        v=(wide.learned_local_cue-wide[method]).to_numpy();m,lo,hi=interval(v)
        ax.scatter(np.arange(len(v))*.008-.076+i,v,s=9,color=COLORS[method],alpha=.45)
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="D",color=COLORS[method],ms=4,capsize=3)
    ax.axhline(0,color="#777777",lw=LW_REF)
    ax.set(xticks=[0,1,2],xticklabels=["vs oracle","vs frozen","vs\nmismatched"],ylabel="Learned minus control (pp)")
    enforce_tokens(fig);fig.savefig(OUT/"figure_S32_panels_A-F.pdf");plt.close(fig)


# Morphology sheet inks: the same two registered hues as the encoder sheet
# (teal = the primary draw, ochre = the contrasting label mode), so the two
# supplementary sheets speak one colour vocabulary.
MORPH_INK={"primary":J.ORDINAL_RAMP[3],"secondary":J.SERIES_COLORS["low_rank"]}


def cell_dots(ax,x,values,color,marker="o",step=.05,s=9,open_=False,zorder=2):
    """Every cell as a small point; tied values are spread evenly side by side."""
    v=np.asarray(values,float);offs=np.zeros(len(v))
    for val in np.unique(np.round(v,9)):
        idx=np.where(np.isclose(v,val,atol=1e-9))[0];k=len(idx)
        offs[idx]=(np.arange(k)-(k-1)/2)*step
    if open_:ax.scatter(x+offs,v,s=s,marker=marker,facecolors="white",edgecolors=color,linewidths=LW_EDGE,alpha=.8,zorder=zorder)
    else:ax.scatter(x+offs,v,s=s,marker=marker,color=color,alpha=.5,edgecolors="none",zorder=zorder)


def cell_series(ax,frame,x,metric,color,label,positions,dodge=0.,paired_lines=False,connect=False,reference=None):
    """Means with 95% cell-bootstrap bars, every cell drawn behind them.

    ``positions`` maps the factor levels to ordinal x positions. With
    ``paired_lines`` the eight cells are joined across levels (a within-cell
    manipulation). ``reference`` names a level whose value is definitional
    (the nominal dictionary compared with itself); it is drawn as an open
    symbol without a bar and is asserted to equal one for every cell.
    """
    values=[];levels=sorted(frame[x].unique())
    wide=frame.pivot(index="root_id",columns=x,values=metric)
    if paired_lines:
        for _,row in wide.iterrows():
            ax.plot([positions[l]+dodge for l in levels],[row[l] for l in levels],color=color,alpha=.3,lw=LW_EDGE,zorder=2)
    for at in levels:
        seeds=wide[at].to_numpy(float);is_ref=(reference is not None and at==reference)
        if is_ref and not np.all(seeds==1):raise ValueError(f"{label} reference level {at} is not identically one")
        cell_dots(ax,positions[at]+dodge,seeds,color,open_=is_ref,zorder=3)
        values.append((positions[at]+dodge,*interval(seeds)))
    values=np.asarray(values)
    ref=np.array([reference is not None and at==reference for at in levels])
    ax.errorbar(values[~ref,0],values[~ref,1],yerr=np.stack([values[~ref,1]-values[~ref,2],values[~ref,3]-values[~ref,1]]),
                color=color,label=label,marker="o",ms=3.5,lw=LW_DATA,capsize=2,ls="-" if connect else "none",
                markeredgecolor="white",markeredgewidth=LW_HAIR,zorder=5)
    if ref.any():ax.scatter(values[ref,0],values[ref,1],s=26,facecolors="white",edgecolors=color,linewidths=LW_DATA,zorder=5)


def centre_grid(fig,axs,margin_pt=6.):
    """Equal outer margins and equal row gutters, measured on the drawn ink.

    The sheet is pasted whole, so its own margins are the page margins: the
    ink is centred left-right, and the rows are re-spaced so that the white
    band between every pair of rows is the same height.
    """
    fw,fh=fig.get_size_inches();margin=margin_pt/72
    for _ in range(3):
        fig.canvas.draw();r=fig.canvas.get_renderer();dpi=fig.dpi
        bb=fig.get_tightbbox(r);dx=((fw-bb.x1)-bb.x0)/2/fw
        rows=[]
        for row in axs:
            centre=np.mean([a.get_position().y0+a.get_position().height/2 for a in row])
            members=[a for a in fig.axes if abs(a.get_position().y0+a.get_position().height/2-centre)<.1]
            boxes=[a.get_tightbbox(r) for a in members]
            rows.append((members,min(b.y0 for b in boxes)/dpi,max(b.y1 for b in boxes)/dpi))
        heights=[y1-y0 for _,y0,y1 in rows];gap=(fh-2*margin-sum(heights))/(len(rows)-1)
        top=fh-margin
        for (members,y0,y1),h in zip(rows,heights):
            dy=(top-y1)/fh
            for a in members:
                pos=a.get_position();a.set_position([pos.x0+dx,pos.y0+dy,pos.width,pos.height])
            top-=h+gap


def morphology():
    source=ROOT/"source_data/review_morphology_uncertainty"
    confusion=pd.read_csv(source/"confusion_pooled.csv").pivot(index="direct_class",columns="proxy_class",values="n_contacts").reindex(index=["E","I"],columns=["E","I"]).fillna(0)
    missing=pd.read_csv(source/"label_missingness.csv")
    cell=pd.read_csv(source/"cell_sensitivity_means.csv")
    comp=pd.read_csv(source/"compression_geometry_audit.csv")
    fig,axs=plt.subplots(3,2,figsize=(FIG_W,7.0))
    fig.subplots_adjust(left=.115,right=.98,bottom=.075,top=.94,wspace=.38,hspace=.68)
    ax=axs[0,0];panel(ax,"A","Direct / proxy label disagreement")
    fraction=confusion.to_numpy()/confusion.sum(axis=1).to_numpy()[:,None]
    im=ax.imshow(100*fraction,vmin=0,vmax=100,cmap="Blues",aspect="auto")
    for i in range(2):
        for j in range(2):ax.text(j,i,f"{int(confusion.iloc[i,j]):,}\n({fraction[i,j]*100:.1f}%)",ha="center",va="center",color="white" if fraction[i,j]>.6 else "black")
    ax.set(xticks=[0,1],xticklabels=["E","I"],yticks=[0,1],yticklabels=["E","I"],xlabel="Target-proxy label",ylabel="Direct label")
    cb=fig.colorbar(im,ax=ax,fraction=.06,pad=.05,ticks=[0,50,100]);cb.set_label("Row (%)");cb.outline.set_visible(False)
    ax=axs[0,1];panel(ax,"B","Direct-label coverage is uneven")
    f=missing[missing.grouping.eq("compartment")];labels=[]
    for i,name in enumerate(["soma","internal","terminal"]):
        g=f[f.stratum.eq(name)];v=100*g.direct_fraction.to_numpy();m,lo,hi=interval(v)
        ax.scatter(np.linspace(-.1,.1,len(v))+i,v,s=14,color=MORPH_INK["primary"],alpha=.65)
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="D",color="#333333",capsize=3,ms=4)
        labels.append(f"{name.capitalize()}\n$n$ = {int(g.n_contacts.sum()):,}")
    ax.set(xticks=[0,1,2],xticklabels=labels,ylabel="Directly typed contacts (%)",xlabel="Compartment (pooled contacts)")
    ax=axs[1,0];panel(ax,"C","Mapping and label choices alter routes")
    positions={2:0,5:1,10:2}
    for mode,color,dodge in [("hybrid",MORPH_INK["primary"],-.17),("direct_only",MORPH_INK["secondary"],.17)]:
        f=cell[cell.label_mode.eq(mode)&cell.radius_log_sd.eq(0)&cell.axial_mode.eq("mean_radius")]
        cell_series(ax,f,"mapping_threshold_um","selected_nominal_jaccard",color,mode.replace("_"," "),positions,dodge=dodge,
                    reference=5 if mode=="hybrid" else None)
    ax.annotate("reference dictionary\n(Jaccard = 1 by definition)",xy=(1-.17,1),xytext=(1.05,.72),ha="center",va="top",fontsize=PT_BASE,color="#555555",
                arrowprops={"arrowstyle":"-","color":"#999999","lw":LW_HAIR,"shrinkB":4})
    ax.set(xlabel="Maximum mapping distance (µm)",ylabel="Selected-route Jaccard",ylim=(-.05,1.05),xlim=(-.5,2.5))
    ax.set_xticks([0,1,2],["2","5","10"]);ax.legend(frameon=False,loc="center",bbox_to_anchor=(.5,.42))
    ax=axs[1,1];panel(ax,"D","Radius sensitivity on a fixed probe")
    positions={0:0,.25:1,.5:2}
    for mode,color,dodge in [("hybrid",MORPH_INK["primary"],-.06),("direct_only",MORPH_INK["secondary"],.06)]:
        f=cell[cell.label_mode.eq(mode)&cell.mapping_threshold_um.eq(5)&cell.axial_mode.eq("mean_radius")]
        cell_series(ax,f,"radius_log_sd","fixed_nominal_field_capture",color,mode.replace("_"," "),positions,dodge=dodge,paired_lines=True)
    ax.set(xlabel="Log-radius perturbation SD",ylabel="Nominal-field capture",ylim=(.05,.6),xlim=(-.5,2.5))
    ax.set_xticks([0,1,2],["0","0.25","0.5"]);ax.legend(frameon=False,loc="upper right")
    ax=axs[2,0];panel(ax,"E","Axial resistance after compression")
    # Ratio on a log x axis (0.1-log-unit bins) and a log count axis: the
    # dominant equal-resistance bin no longer hides the 0.003-0.9 tail.
    ratio=np.log10((1+comp.relative_axial_resistance_error).clip(lower=1e-6))
    if ratio.max()>1e-9:raise ValueError("a compressed segment exceeds its series resistance; the annotation assumes none does")
    edges=np.arange(np.floor(ratio.min()*10)/10,.1+1e-9,.1)
    ax.hist(10**ratio,bins=10**edges,color="#808080",edgecolor="white",lw=LW_HAIR)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_ylim(.7,1500);ax.set_xlim(10**(edges[0]-.05),10**(edges[-1]+.05))
    ax.set_xticks([1e-3,1e-2,1e-1,1],["0.001","0.01","0.1","1"]);ax.set_yticks([1,10,100,1000],["1","10","100","1,000"])
    ax.tick_params(which="minor",width=LW_HAIR,length=1.8)
    tail=int((ratio<-.05).sum())
    ax.text(.03,.95,f"{tail} of {len(ratio)} segments ({comp.root_id.nunique()} cells)\nbelow 0.89 (0.05 log units);\nworst case {10**ratio.min():.3f}",
            transform=ax.transAxes,ha="left",va="top",fontsize=PT_BASE,color="#555555")
    ax.set(xlabel="Compressed / series axial resistance",ylabel="Cable segments")
    ax=axs[2,1];panel(ax,"F","Series-resistance sensitivity")
    f=cell[cell.label_mode.eq("hybrid")&cell.mapping_threshold_um.eq(5)&cell.radius_log_sd.eq(0)]
    wide=f.pivot(index="root_id",columns="axial_mode",values="fixed_nominal_field_capture")
    diff=wide.series_resistance-wide.mean_radius;mean=(wide.series_resistance+wide.mean_radius)/2
    ax.axhline(0,color="#999999",ls="--",lw=LW_REF,zorder=1)
    ax.scatter(mean,diff,s=23,color=MORPH_INK["primary"],edgecolor="white",linewidth=LW_HAIR,zorder=3)
    same=int(np.isclose(diff,0,atol=1e-9).sum())
    for x_,d_ in zip(mean[~np.isclose(diff,0,atol=1e-9)],diff[~np.isclose(diff,0,atol=1e-9)]):
        ax.annotate(f"{d_:+.3f}",xy=(x_,d_),xytext=(4,0),textcoords="offset points",ha="left",va="center",fontsize=PT_BASE,color="#555555")
    ax.text(.03,.05,f"{same} of {len(diff)} cells identical",transform=ax.transAxes,ha="left",va="bottom",fontsize=PT_BASE,color="#555555")
    ax.set(xlabel="Mean of the two captures",ylabel="Series-resistance minus\nmean-radius capture",xlim=(.1,.6),ylim=(-.06,.06))
    ax.set_yticks([-.06,-.03,0,.03,.06])
    centre_grid(fig,axs,margin_pt=6.)
    enforce_tokens(fig);fig.savefig(OUT/"figure_S33_panels_A-F.pdf");plt.close(fig)


if __name__=="__main__":
    style();OUT.mkdir(parents=True,exist_ok=True);encoder();morphology()
