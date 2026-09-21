#!/usr/bin/env python3
"""Read-only independent verification of completed bridge checkpoints."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run
import models

def main():
    cfg=run.freeze(); records=[]; maximum=0.; sign_max=0.; n_pairs=0
    for phase in ('development','fresh'):
        for seed in cfg[phase+'_seeds']:
            pair=[]
            for model in ('algebraic','conductance'):
                folder=run.OUT/'runs'/phase/model
                audit=json.loads((folder/f'seed_{seed}_audit.json').read_text())
                for name,digest in audit['source_files_sha256'].items():
                    assert run.sha(folder/name)==digest
                curves=pd.read_csv(folder/f'seed_{seed}_curves.csv')
                diag=pd.read_csv(folder/f'seed_{seed}_diagnostics.csv')
                assert np.isfinite(curves[['test_nmse','validation_nmse','population_nmse']]).all().all()
                tasks=models.FAMILIES if model=='algebraic' else ['0','1','2']
                for task in tasks:
                    file=folder/f'seed_{seed}_task_{task}'
                    arr=np.load(str(file)+'_states.npz')
                    meta=json.loads(Path(str(file)+'_metadata.json').read_text())
                    assert list(arr['steps'])==cfg['checkpoints']
                    theta=arr['theta']; profile=arr['initial_profiles']; xx=arr['calibration_inputs']
                    if model=='algebraic':
                        initial_state=models.algebra_state(theta[0],xx,arr['left'],arr['right'])
                        xx,yy=run.algebra_data(seed,arr['coefficients'],'test',cfg['algebraic']['n_test'])
                        pred=models.algebra_state(theta[-1],xx,arr['left'],arr['right'])['output']
                        variance=float(arr['coefficients']@arr['coefficients'])
                        if task in ('matching','quartet'):
                            pair.append((theta[0].copy(),arr['left'].copy(),arr['right'].copy(),profile.copy(),arr['calibration_inputs'].copy()))
                    else:
                        initial_state=models.conductance.forward(theta[0],xx,arr['groups'])
                        xx,yy=models.conductance.dataset(seed,int(task),'test',cfg['conductance']['n_test'])
                        pred=models.conductance.forward(theta[-1],xx,arr['groups'])['output']
                        variance=float(np.var(yy))
                        for optimizer in cfg['optimizers']:
                            u=[i for i,r in enumerate(meta['records']) if r['optimizer']==optimizer and r['rule']=='unit_broadcast']
                            s=[i for i,r in enumerate(meta['records']) if r['optimizer']==optimizer and r['rule']=='sign_broadcast']
                            assert [meta['records'][i]['rate'] for i in u]==[meta['records'][i]['rate'] for i in s]
                            difference=float(abs(theta[:,u]-theta[:,s]).max()); sign_max=max(sign_max,difference)
                            assert difference==0., 'Positive-model sign/unit identity failed'
                    np.testing.assert_allclose(initial_state['path'][:,:,:6].mean(axis=1),profile,rtol=0.,atol=0.)
                    errors=np.mean((pred-yy[None])**2,axis=1)/variance
                    end=curves[(curves.task.astype(str)==str(task))&(curves.step==cfg['steps'])]
                    np.testing.assert_allclose(errors,end.test_nmse,rtol=2e-13,atol=2e-14)
                    maximum=max(maximum,float(abs(errors-end.test_nmse.to_numpy()).max()))
                    records.append(dict(phase=phase,seed=seed,model=model,task=str(task),n_conditions=len(errors),checkpoint_count=len(theta)))
                for prefix in ('credit','path'):
                    assert (diag[prefix+'_uniform_oracle_capture'] >= -1e-12).all()
                    assert (diag[prefix+'_best_rank_one_capture'] <= 1+1e-12).all()
                    for direction in ('uniform','calibrated','sign'):
                        assert (diag[prefix+'_'+direction+'_oracle_capture'] <= diag[prefix+'_best_rank_one_capture']+1e-12).all()
                    np.testing.assert_allclose(diag[prefix+'_full_rank_capture'],1.,atol=1e-14)
            for left,right in zip(*pair): np.testing.assert_array_equal(left,right)
            n_pairs+=1
    destination=run.OUT/'summaries'
    pd.DataFrame(records).to_csv(destination/'checkpoint_audit.csv',index=False)
    run.write(destination/'independent_audit.json',dict(status='passed',n_task_checkpoint_files=len(records),
        n_seed_pairs_verified=n_pairs,largest_recomputed_endpoint_error=maximum,
        conductance_unit_sign_max_checkpoint_difference=sign_max,
        verified=['source hashes','all saved checkpoint steps','matching/quartet initial states and tree identity',
            'label-free mean-profile formula','held-out endpoint NMSE from saved weights',
            'conductance sign/unit trajectory identity','all oracle capture bounds','all full-rank baselines']))
    print(json.dumps(json.loads((destination/'independent_audit.json').read_text()),indent=2))

if __name__=='__main__': main()
