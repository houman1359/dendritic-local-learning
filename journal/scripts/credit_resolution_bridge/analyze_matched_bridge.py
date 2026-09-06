#!/usr/bin/env python3
"""Independent spatial-capacity audit of the saved, newly matched learning bridge.

The learning protocol and output are owned by credit_rule_bridge. This file only
reads them. Exact and all restricted learning trajectories are retained; each
candidate dictionary is evaluated at the same reference checkpoint. Both
selected-rate and common-rate conditions are retained with explicit flags.
"""
import json
import numpy as np
import pandas as pd
import analysis_capture as a


def decode_tree(children):
    children={int(k):tuple(v) for k,v in children.items()};children=dict(sorted(children.items()))
    descendants={k:(k,) for k in range(8)};parent={}
    for node,(left,right) in children.items():
        descendants[node]=descendants[left]+descendants[right]
        parent[left]=(node,0);parent[right]=(node,1)
    return a.credit.Tree('matched_compatible','matched',descendants[14],children,descendants,parent,14)


def main():
    base=a.JOURNAL/'source_data/credit_rule_bridge'
    cfg=json.loads((base/'protocol_freeze.json').read_text())['protocol']
    folder=base/'runs/fresh/algebraic';rows=[];spectra=[];eigen=[];gauge=[];checks=[];pairing=[]
    for seed in cfg['fresh_seeds']:
        audit=json.loads((folder/f'seed_{seed}_audit.json').read_text())
        for name,digest in audit['source_files_sha256'].items():assert a.sha(folder/name)==digest
        own=pd.read_csv(folder/f'seed_{seed}_diagnostics.csv')
        paired=[]
        for family in a.credit.FAMILIES:
            path=folder/f'seed_{seed}_task_{family}_states.npz';states=np.load(path)
            meta=json.loads(path.with_name(path.name.replace('_states.npz','_metadata.json')).read_text())
            coeff=states['coefficients'];tree=decode_tree(meta['children']);initial=states['theta'][0,0]
            if family in ('matching','quartet'):paired.append((states['theta'][0],states['left'],states['right'],states['diagnostic_inputs'],states['initial_profiles'],meta['input_spectrum']))
            np.testing.assert_array_equal(states['diagnostic_inputs'],a.credit.domain())
            for j,step in enumerate(states['steps']):
                for i,record in enumerate(meta['records']):
                    metadata=dict(cohort='matched_fresh_algebraic',seed=seed,family=family,step=int(step),structure='compatible',**record)
                    r,s,e=a.analyze_algebraic_state(states['theta'][j,i],tree,coeff,initial,metadata,states['initial_profiles'][i])
                    rows.extend(r);spectra.extend(s);eigen.extend(e)
                    raw=own[(own.task==family)&(own.step==step)&(own.optimizer==record['optimizer'])&(own.rule==record['rule'])&np.isclose(own.rate,record['rate'])].iloc[0]
                    values={v['dictionary']:v for v in r}
                    for ours,theirs in [('uniform_projection','path_uniform_oracle_capture'),('initial_mean_projection','path_calibrated_oracle_capture'),('best_fixed_rank1_q','path_best_rank_one_capture')]:
                        diff=abs(values[ours]['field_energy_fidelity']-raw[theirs]);assert diff<2e-10
                        checks.append(dict(seed=seed,family=family,step=int(step),**record,metric=theirs,difference=diff))
                    diff=abs(values['best_fixed_rank1_error']['error_weighted_energy_fidelity']-raw['credit_best_rank_one_capture']);assert diff<2e-10
                    checks.append(dict(seed=seed,family=family,step=int(step),**record,metric='credit_best_rank_one_capture',difference=diff))
                    _,q,_=a.algebraic_fields(states['theta'][j,i][None],states['left'][i:i+1],states['right'][i:i+1],states['diagnostic_inputs'])
                    q=q[0,:,:6];q/=np.sqrt(np.mean(q*q,axis=0));_,sp=a.spectrum(q)
                    gauge.append(metadata|sp)
        for label,idx in [('initial_parameters',0),('left_children',1),('right_children',2),('evaluation_inputs',3),('initial_profiles',4),('input_spectrum',5)]:
            np.testing.assert_array_equal(paired[0][idx],paired[1][idx])
            pairing.append(dict(seed=seed,quantity=label,matching_quartet_identical=True))
        print('matched capture',seed,'complete',flush=True)
    for label,values in [('capture',rows),('spectra',spectra),('eigenvalues',eigen),('gauge_spectra',gauge),('metric_checks',checks),('pairing_checks',pairing)]:
        pd.DataFrame(values).to_csv(a.OUT/f'matched_bridge_{label}.csv',index=False)
    frame=pd.DataFrame(rows);spec=pd.DataFrame(spectra)
    frame.groupby(['family','optimizer','rule','rate','selected_rate','common_rate','step','dictionary'])[['field_energy_fidelity','error_weighted_energy_fidelity','eligibility_weighted_update_fidelity','population_gradient_cosine_nonroot','population_nmse']].mean().reset_index().to_csv(a.OUT/'matched_bridge_capture_summary.csv',index=False)
    a.dump(a.OUT/'matched_bridge_validation.json',dict(status='passed',seed_blocks=len(cfg['fresh_seeds']),states=len(gauge),metric_checks=len(checks),
        maximum_metric_difference=max(v['difference'] for v in checks),pairing_checks=len(pairing),
        source_protocol_sha256=a.sha(base/'protocol_freeze.json'),source_selection_sha256=a.sha(base/'selection_freeze.json'),script_sha256=a.sha(__file__)))
    # Separate artifact: do not overwrite the archived-cohort diagnostic.
    dest=a.OUT/'matched_bridge';dest.mkdir(exist_ok=True)
    previous=a.OUT
    try:
        a.OUT=dest
        a.plot(frame[frame.selected_rate],spec[spec.selected_rate])
    finally:a.OUT=previous


if __name__=='__main__':main()
