#!/usr/bin/env python3
"""Native figures for new coefficient and morphology-uncertainty analyses."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from journal_style import apply_neurips_style
from neurips_style import FIG_W

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures/supplementary"
COLORS={"oracle_context":"#333333","learned_local_cue":"#16817a",
        "frozen_profile":"#a57422","mismatched_encoder":"#9b4b6d","hard":"#376cb0"}
LABELS={"oracle_context":"Oracle context","learned_local_cue":"Learned, soft",
        "frozen_profile":"Frozen profile","mismatched_encoder":"Mismatched"}


def style():
    apply_neurips_style()
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,
        "axes.labelsize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5,"legend.fontsize":7,
        "pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})


def panel(ax,letter,title):
    ax.set_title(title,loc="left",pad=10)
    ax.text(-.16,1.09,letter,transform=ax.transAxes,fontweight="bold",fontsize=11)


def interval(v):
    v=np.asarray(v,float);rng=np.random.default_rng(83710)
    draw=v[rng.integers(len(v),size=(5000,len(v)))].mean(1)
    return v.mean(),*np.quantile(draw,[.025,.975])


def curve(ax,frame,x,metric,color,label,marker="o",ls="-"):
    values=[]
    for at,g in frame.groupby(x):values.append((at,*interval(g[metric])))
    values=np.asarray(values)
    ax.errorbar(values[:,0],values[:,1],yerr=np.stack([values[:,1]-values[:,2],values[:,3]-values[:,1]]),
                color=color,label=label,marker=marker,ms=3.5,lw=1.25,capsize=2,ls=ls)


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
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",transform=ax.transAxes,fontsize=8,
                bbox={"boxstyle":"round,pad=.5","fc":"#f0f5f4","ec":"#a1b5b2"})
    ax.add_patch(FancyArrowPatch((.5,.64),(.5,.51),transform=ax.transAxes,arrowstyle="->",mutation_scale=11,color="#405a55"))
    ax.text(.5,-.04,"Soft coefficients → four subtree profiles\nMultiply by scalar error and local eligibility",ha="center",va="center",transform=ax.transAxes,fontsize=8)
    ax=axs[0,1];panel(ax,"B","Cue noise limits learning")
    f=final[final.calibration_samples.eq(256)&final.cue_delay_trials.eq(0)]
    for method in LABELS:curve(ax,f[f.method.eq(method)],"cue_noise_sd","accuracy_percent",COLORS[method],LABELS[method])
    h=hfinal[hfinal.calibration_samples.eq(256)&hfinal.cue_delay_trials.eq(0)&hfinal.method.eq("learned_local_cue")]
    curve(ax,h,"cue_noise_sd","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--")
    ax.set(xlabel="Cue noise SD",ylabel="Held-out accuracy (%)",ylim=(10,90))
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc="upper center",bbox_to_anchor=(.54,.995),ncol=3,columnspacing=1.5,handlelength=1.5)
    ax=axs[1,0];panel(ax,"C","Calibration data and computation")
    f=final[final.cue_noise_sd.eq(.5)&final.cue_delay_trials.eq(0)]
    for method in LABELS:curve(ax,f[f.method.eq(method)],"calibration_samples","accuracy_percent",COLORS[method],LABELS[method])
    h=hfinal[hfinal.cue_noise_sd.eq(.5)&hfinal.cue_delay_trials.eq(0)&hfinal.method.eq("learned_local_cue")]
    curve(ax,h,"calibration_samples","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--")
    ax.set(xscale="log",xlabel="Calibration examples (30 epochs)",ylabel="Held-out accuracy (%)",ylim=(10,90));ax.set_xticks([16,64,256],["16","64","256"])
    ax=axs[1,1];panel(ax,"D","Cue / eligibility timing mismatch")
    f=final[final.calibration_samples.eq(256)&final.cue_noise_sd.eq(.5)]
    for method in LABELS:curve(ax,f[f.method.eq(method)],"cue_delay_trials","accuracy_percent",COLORS[method],LABELS[method])
    h=hfinal[hfinal.calibration_samples.eq(256)&hfinal.cue_noise_sd.eq(.5)&hfinal.method.eq("learned_local_cue")]
    curve(ax,h,"cue_delay_trials","accuracy_percent",COLORS["hard"],"Learned, hard*",marker="s",ls="--")
    ax.set(xlabel="Cue delay (independent trials)",ylabel="Held-out accuracy (%)",ylim=(10,90),xticks=[0,1,4])
    ax=axs[2,0];panel(ax,"E","Task-learning trajectories")
    f=soft[soft.calibration_samples.eq(256)&soft.cue_noise_sd.eq(.5)&soft.cue_delay_trials.eq(0)]
    for method in LABELS:curve(ax,f[f.method.eq(method)],"epoch","accuracy_percent",COLORS[method],LABELS[method])
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
    ax.axhline(0,color="#777777",lw=.8)
    ax.set(xticks=[0,1,2],xticklabels=["vs oracle","vs frozen","vs\nmismatched"],ylabel="Learned minus control (pp)")
    fig.savefig(OUT/"figure_S32_panels_A-F.pdf");plt.close(fig)


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
    ax.imshow(fraction,vmin=0,vmax=1,cmap="Blues",aspect="auto")
    for i in range(2):
        for j in range(2):ax.text(j,i,f"{int(confusion.iloc[i,j]):,}\n({fraction[i,j]*100:.1f}%)",ha="center",va="center",color="white" if fraction[i,j]>.6 else "black")
    ax.set(xticks=[0,1],xticklabels=["E proxy","I proxy"],yticks=[0,1],yticklabels=["E direct","I direct"])
    ax=axs[0,1];panel(ax,"B","Direct-label coverage is uneven")
    f=missing[missing.grouping.eq("compartment")]
    for i,name in enumerate(["soma","internal","terminal"]):
        v=100*f[f.stratum.eq(name)].direct_fraction.to_numpy();m,lo,hi=interval(v)
        ax.scatter(np.linspace(-.1,.1,len(v))+i,v,s=14,color="#16817a",alpha=.65)
        ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt="D",color="#333333",capsize=3,ms=4)
    ax.set(xticks=[0,1,2],xticklabels=["Soma","Internal","Terminal"],ylabel="Directly typed contacts (%)")
    ax=axs[1,0];panel(ax,"C","Mapping and label choices alter routes")
    for mode,color in [("hybrid","#16817a"),("direct_only","#a57422")]:
        f=cell[cell.label_mode.eq(mode)&cell.radius_log_sd.eq(0)&cell.axial_mode.eq("mean_radius")]
        curve(ax,f,"mapping_threshold_um","selected_nominal_jaccard",color,mode.replace("_"," "))
    ax.set(xlabel="Maximum mapping distance (µm)",ylabel="Selected-route Jaccard",ylim=(-.05,1.05),xticks=[2,5,10]);ax.legend(frameon=False,loc="best")
    ax=axs[1,1];panel(ax,"D","Radius sensitivity on a fixed probe")
    for mode,color in [("hybrid","#16817a"),("direct_only","#a57422")]:
        f=cell[cell.label_mode.eq(mode)&cell.mapping_threshold_um.eq(5)&cell.axial_mode.eq("mean_radius")]
        curve(ax,f,"radius_log_sd","fixed_nominal_field_capture",color,mode.replace("_"," "))
    ax.set(xlabel="Log-radius perturbation SD",ylabel="Nominal-field capture",ylim=(0,1.05),xticks=[0,.25,.5])
    ax=axs[2,0];panel(ax,"E","Axial resistance after compression")
    ratio=1+comp.relative_axial_resistance_error
    ax.hist(np.log10(ratio.clip(lower=1e-6)),bins=25,color="#808080",edgecolor="white")
    ax.axvline(0,color="#222222",lw=1)
    ax.set(xlabel="log₁₀(compressed / series resistance)",ylabel="Cable segments")
    ax=axs[2,1];panel(ax,"F","Series-resistance sensitivity")
    f=cell[cell.label_mode.eq("hybrid")&cell.mapping_threshold_um.eq(5)&cell.radius_log_sd.eq(0)]
    wide=f.pivot(index="root_id",columns="axial_mode",values="fixed_nominal_field_capture")
    ax.plot([0,1],[0,1],color="#999999",ls="--",lw=.8)
    ax.scatter(wide.mean_radius,wide.series_resistance,s=23,color="#16817a",edgecolor="white",linewidth=.4)
    ax.set(xlabel="Mean-radius field capture",ylabel="Series-resistance\nfield capture",xlim=(0,1),ylim=(0,1))
    fig.savefig(OUT/"figure_S33_panels_A-F.pdf");plt.close(fig)


if __name__=="__main__":
    style();OUT.mkdir(parents=True,exist_ok=True);encoder();morphology()
