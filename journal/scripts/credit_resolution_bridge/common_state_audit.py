#!/usr/bin/env python3
"""Counterfactual labels on the exact same tree, weights and full input domain.

This changes labels for evaluation only; it does not retrain or imply the
counterfactual target has been learned. A companion fully matched learning
cohort handles that intervention. All six actual nonroot internal sites used.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import analysis_capture as a


def main():
    rows=[];spectra=[];eigen=[];checks=[]
    for path in sorted((a.OUT/'replay').glob('seed_*_state_metadata.json')):
        seed=int(path.name.split('_')[1]);meta=json.loads(path.read_text());entry=next(v for v in meta if v['family']=='matching')
        coeff=np.array(entry['coefficient']);tree,_,_=a.credit.tree_from_coeff(coeff,'common_balanced')
        quartet=np.zeros(256)
        for index,child in enumerate(tree.children[14]):
            support=tree.descendants[child];assert len(support)==4
            quartet[sum(1<<leaf for leaf in support)]=.5*(-1 if index else 1)
        cov_diff=float(np.max(abs(a.credit.input_gradient_covariance(coeff)-a.credit.input_gradient_covariance(quartet))))
        assert cov_diff<1e-13
        assert a.credit.cut_scores(quartet,tree)['centered_cut_bound']<1e-13
        weights=np.load(a.OUT/'replay'/f'seed_{seed}_checkpoints.npz')['matching'];initial=np.array(entry['initial'])
        for i,m in enumerate(entry['metadata']):
            if m['rule']!='exact':continue
            for j,step in [(0,0),(5,1024)]:
                same_state=[]
                for name,target in [('matching',coeff),('quartet',quartet)]:
                    metadata=dict(cohort='counterfactual_same_state',seed=seed,family=name,optimizer=m['optimizer'],
                        rule='reference_matching_exact',rate=m['rate'],step=step,source_state='matching_exact_trained' if step else 'same_initial')
                    r,s,e=a.analyze_algebraic_state(weights[j,i],tree,target,initial,metadata)
                    rows.extend(r);spectra.extend(s);eigen.extend(e);same_state.append(pd.DataFrame(r).set_index('dictionary'))
                difference=np.max(abs(same_state[0].field_energy_fidelity-same_state[1].field_energy_fidelity))
                unweighted=[d for d in same_state[0].index if d!='best_fixed_rank1_error']
                difference=np.max(abs(same_state[0].loc[unweighted,'field_energy_fidelity']-same_state[1].loc[unweighted,'field_energy_fidelity']))
                # The residual-optimized profile is explicitly target-dependent, so excluded.
                assert difference<1e-13
                checks.append(dict(seed=seed,optimizer=m['optimizer'],step=step,input_covariance_max_difference=cov_diff,
                    unweighted_common_dictionary_capture_max_difference=float(difference),compatible_both=True))
    for name,data in [('capture',rows),('spectra',spectra),('eigenvalues',eigen),('checks',checks)]:
        pd.DataFrame(data).to_csv(a.OUT/f'counterfactual_same_state_{name}.csv',index=False)
    a.dump(a.OUT/'counterfactual_same_state_validation.json',dict(status='passed',state_pairs=len(checks),
        maximum_unweighted_capture_difference=max(r['unweighted_common_dictionary_capture_max_difference'] for r in checks),
        interpretation='Unweighted q capture identical under relabeling of the exact same state; error weighting and its separately optimized profile may change.',
        script_sha256=a.sha(__file__)))


if __name__=='__main__':main()
