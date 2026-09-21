#!/usr/bin/env python3
"""Sensitivity of credit spectra to static unit scaling.

Each spatial coordinate is divided by its RMS output sensitivity. This is a
post hoc geometry diagnostic, not a reparameterized learning experiment. The
resulting spectrum is invariant to invertible diagonal unit rescaling (up to
signs, which preserve singular values). Raw energy capture remains gauge-specific.
"""
import json
import numpy as np
import pandas as pd
import analysis_capture as a


def main():
    rows=[];checks=[]
    for path in sorted((a.OUT/'replay').glob('seed_*_state_metadata.json')):
        seed=int(path.name.split('_')[1]);entries=json.loads(path.read_text())
        saved=np.load(a.OUT/'replay'/f'seed_{seed}_checkpoints.npz')
        for entry in entries:
            family=entry['family'];tree,_,_=a.credit.tree_from_coeff(np.array(entry['coefficient']),'reference')
            left=np.array([[tree.children[n][0] for n in range(8,15)]]);right=np.array([[tree.children[n][1] for n in range(8,15)]])
            x=a.credit.domain();y=a.credit.fourier_design(x)@np.array(entry['coefficient'])
            for j,step in enumerate([0,1,16,64,256,1024]):
                weights=saved[family][j];count=len(weights)
                output,q,features=a.algebraic_fields(weights,np.repeat(left,count,axis=0),np.repeat(right,count,axis=0),x)
                for i,m in enumerate(entry['metadata']):
                    field=q[i,:,:6];scale=np.sqrt(np.mean(field**2,axis=0));assert np.all(scale>1e-30)
                    whitened=field/scale
                    transformed=field*np.array([.1,-.5,1.,3.,-7.,11.])
                    transformed/=np.sqrt(np.mean(transformed**2,axis=0))
                    eigen,spec=a.spectrum(whitened);other,_=a.spectrum(transformed)
                    difference=float(np.max(abs(eigen-other)));assert difference<1e-11
                    checks.append(difference)
                    rows.append(dict(seed=seed,family=family,step=step,**m,field='path_q_unit_rms',**spec,
                        interpretation='Column RMS normalized; energy fractions invariant to static invertible per-site scaling'))
    df=pd.DataFrame(rows);df.to_csv(a.OUT/'gauge_normalized_spectra.csv',index=False)
    df.groupby(['family','optimizer','rule','step'])[['effective_rank','rank95','rank99','leading_fraction']].mean().reset_index().to_csv(a.OUT/'gauge_normalized_spectrum_summary.csv',index=False)
    a.dump(a.OUT/'gauge_validation.json',dict(status='passed',states=len(checks),max_scaling_spectrum_difference=max(checks),
        script_sha256=a.sha(__file__),scope='Static coordinate scaling only; affine state shifts/different parameterizations or nonlinear coordinates not covered'))


if __name__=='__main__':main()
