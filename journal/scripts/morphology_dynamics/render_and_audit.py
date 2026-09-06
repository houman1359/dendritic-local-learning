#!/usr/bin/env python3
"""Post-outcome diagnostics and standalone figures; no selector fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import investigate as study


def main():
    source = study.OUTPUT/'runs/fresh'
    dest = study.OUTPUT/'summaries/fresh'
    figures = study.OUTPUT/'figures'; figures.mkdir(exist_ok=True)
    audits = [json.loads(p.read_text()) for p in sorted(source.glob('*_audit.json'))]
    assert [a['seed'] for a in audits] == list(range(9300,9320))
    for audit in audits:
        seed = audit['seed']
        assert study.sha(source/f'seed_{seed}_predictions.csv') == audit['prediction_seal']['predictions_sha256']
    predictions = pd.concat([pd.read_csv(source/f"seed_{a['seed']}_predictions.csv") for a in audits], ignore_index=True)
    outcomes = pd.concat([pd.read_csv(source/f"seed_{a['seed']}_outcomes.csv") for a in audits], ignore_index=True)
    key = ['task_id','arm','candidate_id','checkpoint']
    q = predictions[predictions.method.eq('gaussian_oracle_sgd')]
    comparison = outcomes.merge(q[key+['predicted_loss']], on=key, validate='many_to_one')
    comparison['population_error'] = comparison.predicted_loss-comparison.population_loss
    bias_rows = []
    for group, g in comparison.groupby(['arm','regime','checkpoint','rank','noise_sd']):
        mean, lo, hi = study.bootstrap(g.groupby('seed').population_error.mean())
        bias_rows.append(dict(zip(['arm','regime','checkpoint','rank','noise_sd'],group)) | dict(
            mean_prediction_minus_population_loss=mean, ci95_low=lo, ci95_high=hi,
            n_seed_blocks=g.seed.nunique(), n_candidate_outcomes=len(g)))
    biases = pd.DataFrame(bias_rows)
    biases.to_csv(dest/'oracle_population_bias_by_regime.csv', index=False)
    mean_bias_rows = []
    for group,g in comparison.groupby(['arm','regime','checkpoint']):
        mean, lo, hi = study.bootstrap(g.groupby('seed').population_error.mean())
        mean_bias_rows.append(dict(zip(['arm','regime','checkpoint'],group)) | dict(
            mean_prediction_minus_population_loss=mean,ci95_low=lo,ci95_high=hi,n_seed_blocks=g.seed.nunique()))
    pd.DataFrame(mean_bias_rows).to_csv(dest/'oracle_population_bias_overall.csv', index=False)
    contrasts = []
    for (arm, checkpoint),g in comparison.groupby(['arm','checkpoint']):
        p = g.pivot(index=['task_id','seed','candidate_id'],columns='regime',values='population_error')
        diff = (p.fixed_cache-p.fresh_iid).groupby('seed').mean()
        mean, lo, hi = study.bootstrap(diff)
        contrasts.append(dict(arm=arm,checkpoint=checkpoint,fixed_cache_minus_fresh_prediction_error=mean,
                              ci95_low=lo,ci95_high=hi,n_seed_blocks=len(diff)))
    pd.DataFrame(contrasts).to_csv(dest/'cache_misspecification_contrasts.csv', index=False)
    summary = pd.read_csv(dest/'endpoint_summary_with_secondary_baselines.csv')
    colors = dict(original_scalar='#a54443',rank_only='#827f7a',pilot16='#d29c37',
        empirical_split_fullbatch='#76aeb2',gaussian_plugin_fullbatch='#6075b8',gaussian_plugin_sgd='#136c78',
        observed_count_cheapest='#2d8945',observed_count_conditioning='#6fa87a',gaussian_oracle_sgd='#28282c')
    labels = dict(original_scalar='Original scalar',rank_only='Original rank policy',pilot16='16-step pilot',
        empirical_split_fullbatch='Empirical spectral',gaussian_plugin_fullbatch='Plug-in spectral',
        gaussian_plugin_sgd='Plug-in SGD moments',observed_count_cheapest='Context count + cheapest',
        observed_count_conditioning='Count + conditioning',gaussian_oracle_sgd='Population-oracle SGD')
    methods=list(labels)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes=plt.subplots(2,2,figsize=(13,9.5),layout='constrained')
    for ax,arm,title,letter in zip(axes[0],['feedback_only','joint_forward_feedback'],['Feedback only','Joint forward and feedback'],['A','B']):
        d=summary[summary.arm.eq(arm)&summary.regime.eq('fixed_cache')&summary['rank'].eq('all')].set_index('method')
        for i,method in enumerate(methods):
            r=d.loc[method]
            ax.errorbar(r.mean_regret,i,xerr=[[max(r.mean_regret-r.ci95_low,0)],[max(r.ci95_high-r.mean_regret,0)]],
                        marker='o',markersize=5,color=colors[method],capsize=2)
            ax.text(max(d.mean_regret)*1.05,i,f'{r.mean_regret:.6f}',va='center',fontsize=9,color=colors[method])
        ax.set_yticks(range(len(methods)),[labels[m] for m in methods]);ax.invert_yaxis()
        ax.set_xlim(-.002,max(d.mean_regret)*1.36)
        ax.set_xlabel('Final objective regret (half-MSE + resource cost)')
        ax.set_title(f'{letter}  {title}: original finite training cache',loc='left',weight='bold')
        ax.axvline(0,lw=.6,color='.7');ax.grid(axis='x',alpha=.15)
    ax=axes[1,0]
    strong=['gaussian_plugin_sgd','gaussian_plugin_fullbatch','observed_count_cheapest','gaussian_oracle_sgd']
    for j,method in enumerate(strong):
        d=summary[summary.arm.eq('joint_forward_feedback')&summary.regime.eq('fixed_cache')&summary.method.eq(method)&summary['rank'].ne('all')].copy()
        d['numeric_rank']=d['rank'].astype(int);d=d.sort_values('numeric_rank')
        x=np.arange(4)+(j-1.5)*.07
        ax.errorbar(x,d.mean_regret,yerr=[np.maximum(d.mean_regret-d.ci95_low,0),np.maximum(d.ci95_high-d.mean_regret,0)],
                    marker='o',lw=1,color=colors[method],capsize=2,label=labels[method])
    ax.set_xticks(range(4),[1,2,4,8]);ax.set_xlabel('Generating rank');ax.set_ylabel('Final objective regret')
    ax.set_title('C  Differences among strong selectors (joint arm)',loc='left',weight='bold')
    ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    ax=axes[1,1]
    cost=pd.read_csv(dest/'comparative_computational_cost.csv').set_index('method')
    cost_methods=['observed_count_cheapest','gaussian_plugin_fullbatch','gaussian_plugin_sgd','empirical_split_fullbatch','original_scalar','pilot16','candidate_training']
    cost_labels={**labels,'candidate_training':'All candidates: 256 updates'}
    values=[cost.loc[m,'median_seconds_per_20_candidates'] for m in cost_methods]
    ax.barh(range(len(values)),values,color=[colors.get(m,'#555555') for m in cost_methods],alpha=.85)
    ax.set_yticks(range(len(values)),[cost_labels[m] for m in cost_methods]);ax.invert_yaxis();ax.set_xscale('log')
    ax.set_xlabel('Median CPU seconds per 20-candidate decision')
    ax.set_title('D  Measured implementation cost',loc='left',weight='bold')
    ax.text(0,-.26,'Plug-in forecasts include calibration-model preparation.\nCount-only time excludes shared calibration generation. Pilot includes 16 updates + calibration evaluation.',
            transform=ax.transAxes,fontsize=8,va='top')
    fig.suptitle('Fresh seeds: 320 tasks per arm and training regime; 20 independent seed blocks\nPoints and intervals: means and 95% seed-block bootstrap intervals. Original negative result retained.',fontsize=12)
    fig.savefig(figures/'finite_horizon_selection.pdf',bbox_inches='tight');fig.savefig(figures/'finite_horizon_selection.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig, axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    for col,arm in enumerate(['feedback_only','joint_forward_feedback']):
        actual=outcomes[outcomes.arm.eq(arm)&outcomes.regime.eq('fixed_cache')&outcomes.checkpoint.eq(256)]
        for row,method in enumerate(['original_scalar','gaussian_plugin_sgd']):
            z=actual.merge(predictions[predictions.method.eq(method)][key+['predicted_loss']],on=key,validate='one_to_one')
            ax=axes[row,col]
            hb=ax.hexbin(z.predicted_loss,z.test_loss,gridsize=35,mincnt=1,cmap='viridis',bins='log')
            lo=min(z.predicted_loss.min(),z.test_loss.min());hi=max(z.predicted_loss.max(),z.test_loss.max())
            ax.plot([lo,hi],[lo,hi],color='#d14a4a',lw=1,ls='--')
            ax.set_xlabel('Predicted final half-MSE');ax.set_ylabel('Observed final test half-MSE')
            ax.set_title(f'{chr(65+row*2+col)}  {arm.replace("_"," ")}: {labels[method]}',loc='left',weight='bold')
            fig.colorbar(hb,ax=ax,label='Candidate count (log scale)',fraction=.04,pad=.02)
        ax=axes[2,col]
        for j,regime in enumerate(['fixed_cache','fresh_iid']):
            z=biases[biases.arm.eq(arm)&biases.regime.eq(regime)&biases.checkpoint.eq(256)&biases.noise_sd.eq(.75)].sort_values('rank')
            ax.errorbar(np.arange(4)+(j-.5)*.08,z.mean_prediction_minus_population_loss,
                yerr=[z.mean_prediction_minus_population_loss-z.ci95_low,z.ci95_high-z.mean_prediction_minus_population_loss],
                marker='o',capsize=2,label=regime.replace('_',' '),color=['#ad5c3e','#246d8d'][j])
        ax.axhline(0,color='.4',lw=.7);ax.set_xticks(range(4),[1,2,4,8]);ax.set_xlabel('Generating rank')
        ax.set_ylabel('Oracle prediction − realized population loss')
        ax.set_title(f'{chr(69+col)}  Fresh-example assumption (noise SD 0.75)',loc='left',weight='bold');ax.legend(frameon=False)
    fig.suptitle('Finite-horizon loss forecasts and finite-cache misspecification\nCalibration-derived forecasts; explicitly privileged population oracle for the assumption diagnostic.',fontsize=12)
    fig.savefig(figures/'finite_horizon_prediction_diagnostics.pdf',bbox_inches='tight');fig.savefig(figures/'finite_horizon_prediction_diagnostics.png',dpi=150,bbox_inches='tight');plt.close(fig)
    expected_outcomes=20*16*2*2*20*3
    expected_predictions=20*16*2*320
    nonnegative_methods=predictions[~predictions.method.eq('original_scalar')]
    study.save_json(study.OUTPUT/'fresh_validation.json',dict(n_seed_blocks=len(audits),n_outcomes=len(outcomes),
        expected_outcomes=expected_outcomes,n_predictions=len(predictions),expected_predictions=expected_predictions,
        complete=len(outcomes)==expected_outcomes and len(predictions)==expected_predictions,
        all_prediction_seals_verified=True,original_freeze_verified=bool(study.check_freeze()),
        min_non_scalar_predicted_loss=float(nonnegative_methods.predicted_loss.min()),
        all_outcomes_finite=bool(np.isfinite(outcomes[['test_loss','population_loss']]).all().all()),
        all_outputs_retained=True,figures=[p.name for p in sorted(figures.glob('*.pdf'))]))


if __name__=='__main__':
    main()
