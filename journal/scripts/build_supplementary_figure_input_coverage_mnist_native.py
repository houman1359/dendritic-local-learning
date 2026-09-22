"""Render the verified MNIST spatial-map comparison, excluding historical noise tasks."""
from pathlib import Path
import numpy as np
import pandas as pd
from figure_canvas import NativeCanvas, Margins, COLORS, LW_ERR, LW_REF, MARKER_MS, SEED_MS

ROOT=Path(__file__).resolve().parents[1]
SUMMARY=ROOT/'source_data/spatial_topology_audit/task_feedback_effects.csv'
SEEDS=ROOT/'source_data/prospective_input_validity/followup_publication_seed_outcomes.csv'
ORDER=('backprop','per_soma','per_soma_shared','path_transport')

def plot_data():
    summary=pd.read_csv(SUMMARY).query("task == 'mnist'").set_index('feedback').loc[list(ORDER)]
    runs=pd.read_csv(SEEDS)
    runs=runs[runs.family.eq('spatial') & runs.task.eq('mnist')]
    assert set(runs.core)=={'dendritic_shunting','dendritic_additive'}
    pairs=runs.pivot(index=['core','strategy','feedback','seed'],columns='topology',values='test_accuracy').reset_index()
    pairs['difference']=pairs.spatial-pairs.random
    paired=pairs.groupby(['feedback','seed'],as_index=False).difference.mean()
    assert len(paired)==40 and paired.groupby('feedback').size().eq(10).all()
    for rule in ORDER:
        z=paired[paired.feedback.eq(rule)].difference.to_numpy()
        np.testing.assert_allclose(z.mean(),summary.loc[rule,'mean_difference'],rtol=0,atol=1e-12)
    return summary,paired

def build():
    summary,paired=plot_data()
    c=NativeCanvas(2.0,1,margins=Margins(left=48,right=20,top=22,bottom=34),letter_clearance=True)
    ax=c.panel('A',0,0,12,grid='y')
    rows=[]
    for x,rule in enumerate(ORDER):
        z=paired[paired.feedback.eq(rule)].sort_values('seed')
        ax.scatter(x+np.linspace(-.09,.09,len(z)),100*z.difference,s=SEED_MS**2,color=COLORS['ink'],alpha=.5,zorder=3)
        row=summary.loc[rule];mean=100*row.mean_difference
        ax.errorbar(x,mean,yerr=[[mean-100*row.ci95_low],[100*row.ci95_high-mean]],fmt='D',ms=MARKER_MS,mfc='white',mec=COLORS['ink'],ecolor=COLORS['ink'],lw=LW_ERR,capsize=2,zorder=4)
        rows.extend(dict(panel='D',task='mnist',feedback=rule,record='paired seed difference',seed=int(r.seed),difference=float(r.difference)) for r in z.itertuples())
        rows.append(dict(panel='D',task='mnist',feedback=rule,record='summary',mean_difference=row.mean_difference,ci95_low=row.ci95_low,ci95_high=row.ci95_high,n_seeds=10))
    ax.axhline(0,color=COLORS['mute'],ls='--',lw=LW_REF)
    ax.set_xticks(range(4),['BP','MW scalar','Neuron','Exact path'])
    ax.set_xlim(-.45,3.45);ax.set_ylim(-.15,2.9)
    ax.set_ylabel('Spatial − random accuracy (pp)')
    c.lock_reserves()
    path=ROOT/'figures/supplementary/figure_input_coverage_mnist_native.pdf'
    issues=c.save(path,name='input_coverage_mnist')
    pd.DataFrame(rows).to_csv(ROOT/'source_data/curated_publication/si_mnist_coverage_plotted.csv',index=False)
    for issue in issues:print(issue)
    return issues

if __name__=='__main__':raise SystemExit(bool(build()))
