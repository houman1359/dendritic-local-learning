"""Replace redundant Figure 4 panels with the paired noise sensitivity."""
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import COLORS, style_panel

J=Path(__file__).resolve().parents[2]


def panel(ax, letter, rows, *, common=False):
    d=pd.read_csv(J/'source_data/curated_publication/noise_controls_curves.csv')
    selected=d.common_rate if common else d.selected_rate
    d=d[selected & d.step.eq(16384)]
    summary=pd.read_csv(J/'source_data/curated_publication/noise_controls_contrasts.csv')
    summary=summary[summary.common_rate.eq(common)&summary.step.eq(16384)&summary.metric.eq('population_nmse')]
    style_panel(ax)
    for i,noise in enumerate(['fixed_absolute','noise_free','relative_matched']):
        wide=d[d.noise.eq(noise)].pivot(index='seed',columns=['task','rule'],values='population_nmse')
        v=(wide['quartet','calibrated_broadcast']-wide['quartet','exact'])-(wide['matching','calibrated_broadcast']-wide['matching','exact'])
        r=summary[summary.noise.eq(noise)].iloc[0]
        ax.scatter(i+np.linspace(-.12,.12,len(v)),v,s=5,lw=0,color=COLORS['additive'],alpha=.3)
        ax.errorbar(i,r['mean'],yerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt='D',
                    color=COLORS['additive'],mfc='white',ms=4,lw=.85,capsize=2)
        rows.append(dict(panel=letter,record='summary',series='quartic-minus-pairwise calibrated-minus-exact',noise=noise,
             common_rate=common,mean=r['mean'],ci_low=r.ci_low,ci_high=r.ci_high,n_seeds=20,endpoint='16384 updates; clean complete domain'))
        rows.extend(dict(panel=letter,record='paired seed difference',series='paired seed contrast',noise=noise,common_rate=common,
                         seed=int(seed),value=float(value)) for seed,value in v.items())
    ax.axhline(0,color=COLORS['mute'],ls=(0,(2.2,1.8)),lw=.55)
    ax.set_xlim(-.45,2.45);ax.set_ylim(-.08,1.65);ax.set_yticks([0,.5,1.,1.5])
    ax.set_xticks([0,1,2],['Fixed\nabsolute','Noise\nfree','Relative\nmatched'])
    ax.set_ylabel('Interaction deficit',fontsize=7)
    ax.set_title('Common-rate noise control' if common else 'Noise control, 16,384 updates',fontsize=8,loc='left',pad=10)
    ax.tick_params(labelsize=7)
