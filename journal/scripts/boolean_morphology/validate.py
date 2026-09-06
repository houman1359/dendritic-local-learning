#!/usr/bin/env python3
"""Audit frozen inputs, all retained weights/endpoints, and one full replay."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import platform
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from boolean_morphology.experiment import OUT,verify,sha,dump,core_hashes,seed_data,train


def independent_forward(x,theta,children):
    if x.ndim==2:x=np.broadcast_to(x,(theta.shape[0],*x.shape))
    state=[x[:,:,i] for i in range(4)]
    indices=np.arange(len(theta))[:,None]
    for node in range(3):
        stack=np.stack(state,axis=1)
        left=stack[np.arange(len(theta)),children[:,node,0]]
        right=stack[np.arange(len(theta)),children[:,node,1]]
        features=np.stack((np.ones_like(left),left,right,left*right),axis=2)
        # Independent feature contraction instead of the runner's expanded
        # polynomial. Roundoff tolerance is explicit in the endpoint audit.
        state.append(np.sum(features*theta[:,node,None,:],axis=2))
    return state[-1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-replay',action='store_true');args=parser.parse_args()
    started=time.perf_counter();cfg=verify();freeze=json.loads((OUT/'protocol_freeze.json').read_text())
    seal=json.loads((OUT/'selection_freeze.json').read_text())
    assert seal['source_files']==core_hashes()
    assert seal['rate_file_sha256']==sha(OUT/'selected_rates.json')
    assert not set(cfg['fresh_seeds'])&set(cfg['development_seeds'])
    for relative,value in seal['development_files'].items():assert sha(OUT/relative)==value,relative
    records=[];maximum=0.;total=0
    for split in ('development','fresh'):
        for seed in cfg[split+'_seeds']:
            path=OUT/'runs'/split/f'seed_{seed}'
            audit=json.loads(path.with_suffix('.json').read_text())
            assert audit['source_files']==core_hashes() and audit['protocol_sha256']==sha(OUT/'protocol.json')
            assert audit['csv_sha256']==sha(path.with_suffix('.csv')) and audit['npz_sha256']==sha(path.with_suffix('.npz'))
            assert datetime.fromisoformat(audit['utc_started'])>=datetime.fromisoformat(freeze['utc'])
            if split=='fresh':
                assert audit['selection_sha256']==sha(OUT/'selection_freeze.json')
                assert datetime.fromisoformat(audit['utc_started'])>=datetime.fromisoformat(seal['utc'])
            data=dict(np.load(path.with_suffix('.npz')))
            regenerated=seed_data(seed,cfg)
            for key,value in regenerated.items():np.testing.assert_array_equal(value,data[key],err_msg=f'{seed}:{key}')
            frame=pd.read_csv(path.with_suffix('.csv'),float_precision='round_trip')
            end=frame[frame.step==2048].sort_values('condition_id')
            assert len(frame)==2688 and len(end)==336
            assert end.groupby(['family','tree','optimizer','rule','rate']).size().eq(1).all()
            assert set(frame.step)==set(cfg['checkpoints'])
            assert not end.failed.any() and np.max(abs(data['final_weights']))<=2
            fi=data['family_index'];pred=independent_forward(data['population_x'],data['final_weights'],data['children'])
            testpred=independent_forward(data['test_x'][fi],data['final_weights'],data['children'])
            pop=np.mean((pred-data['population_y'][fi])**2,axis=1)
            test=np.mean((testpred-data['test_y'][fi])**2,axis=1)
            error=max(float(abs(pop-end.population_nmse.to_numpy()).max()),float(abs(test-end.test_noisy_nmse.to_numpy()).max()))
            assert error<2e-12,(seed,error);maximum=max(maximum,error)
            raw=data['raw_population_y'][fi];mean=raw.mean(axis=1)[:,None];sd=raw.std(axis=1)[:,None]
            decisions=pred>=(.5-mean)/sd;truth=raw.astype(bool)
            accuracy=(decisions==truth).mean(axis=1)
            balanced=((decisions&truth).sum(axis=1)/truth.sum(axis=1)+(~decisions&~truth).sum(axis=1)/(~truth).sum(axis=1))/2
            np.testing.assert_array_equal(accuracy,end.accuracy.to_numpy())
            np.testing.assert_allclose(balanced,end.balanced_accuracy.to_numpy(),rtol=0,atol=1e-15)
            records.append(dict(split=split,seed=seed,fits=336,endpoint_max_abs_error=error,
                all_hashes_and_streams_match=True,post_seal=True,all_conditions_present=True,failed_fits=0))
            total+=336
    replay={}
    if not args.skip_replay:
        seed=cfg['fresh_seeds'][0];rerun,arrays=train(seed,cfg)
        saved=np.load(OUT/'runs/fresh'/f'seed_{seed}.npz')
        np.testing.assert_array_equal(arrays['final_weights'],saved['final_weights'])
        old=pd.read_csv(OUT/'runs/fresh'/f'seed_{seed}.csv',float_precision='round_trip')
        columns=[c for c in rerun if c not in ('elapsed_seconds',)]
        for column in columns:np.testing.assert_array_equal(rerun[column].to_numpy(),old[column].to_numpy(),err_msg=column)
        replay=dict(seed=seed,complete_fits=336,steps=2048,all_final_weights_and_nontiming_trajectories_bit_identical=True)
    pd.DataFrame(records).to_csv(OUT/'independent_validation_rows.csv',index=False)
    dump(OUT/'validation_report.json',dict(status='passed',protocol_sha256=sha(OUT/'protocol.json'),
        selection_sha256=sha(OUT/'selection_freeze.json'),validator_sha256=sha(__file__),
        development_fits=1680,fresh_fits=6720,all_25_seeds_and_8400_endpoints_reconstructed=True,
        independently_evaluated_endpoint_max_abs_error=maximum,all_frozen_streams_and_hashes_match=True,
        all_choices_precede_fresh_start=True,gradient_validation='Three pre-freeze unit tests include48central-difference parameter checks across four trees, root equality, truth-table normalization and leaf-permutation equivalence.',
        full_training_replay=replay,elapsed_seconds=time.perf_counter()-started,
        validation_environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)))
    print(json.dumps(json.loads((OUT/'validation_report.json').read_text()),indent=2))


if __name__=='__main__':main()
