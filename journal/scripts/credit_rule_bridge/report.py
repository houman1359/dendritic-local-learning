#!/usr/bin/env python3
"""Summarize the frozen bridge without altering any training result."""
import pandas as pd
import run

def main():
    folder=run.OUT/'summaries'
    curves=pd.read_csv(folder/'all_curves.csv')
    endpoint=curves[(curves.step==1024)&curves.selected_rate]
    rows=[]
    for optimizer,part in endpoint[endpoint.model=='conductance'].groupby('optimizer'):
        wide=part.groupby(['seed','rule']).test_nmse.mean().unstack()
        for rule in ['unit_broadcast','calibrated_broadcast','sign_broadcast']:
            rows.append(dict(optimizer=optimizer,rule=rule,mean_rule_nmse=float(wide[rule].mean()),
                mean_exact_nmse=float(wide.exact.mean()),**run.bootstrap(wide[rule]-wide.exact)))
    pd.DataFrame(rows).to_csv(folder/'conductance_seed_averaged_contrasts.csv',index=False)
    columns=['model','task','optimizer','rule','step','path_uniform_oracle_capture','path_calibrated_oracle_capture',
             'path_best_rank_one_capture','path_effective_rank','credit_uniform_oracle_capture',
             'credit_calibrated_oracle_capture','credit_best_rank_one_capture','credit_effective_rank']
    geometry=pd.read_csv(folder/'selected_credit_geometry.csv')
    geometry[(geometry.rule=='exact')&(geometry.optimizer=='adam')][columns].to_csv(folder/'exact_adam_credit_geometry.csv',index=False)
    endpoint.groupby(['model','optimizer','rule'])[['gradient_clipped_steps','parameter_projected_steps']].agg(['mean','max']).to_csv(folder/'optimization_bound_events.csv')
    pd.DataFrame([dict(model='algebraic',task=task,clean_target_variance=variance,label_noise_variance=.15**2,
        expected_label_noise_nmse=.15**2/variance,training_half_mse_gradient_denominator=variance,
        population_nmse_denominator=variance,test_nmse_denominator=variance)
        for task,variance in [('matching',1.),('quartet',.5),('nested',1.)]]).to_csv(folder/'task_variance_and_noise_floor.csv',index=False)

if __name__=='__main__': main()
