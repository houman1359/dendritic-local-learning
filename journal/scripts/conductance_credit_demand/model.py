"""Directed seven-compartment E/I tree; production equation, no new nonlinearity.

All 16 positive conductances are trained.  The existing reference model uses
identity reactivation, E_E=1, E_I=E_L=0 and leak=1.  Child coupling is directed,
not reciprocal cable transport.  Context only changes externally supplied
inhibitory activity; it is not delivered to the credit rule.
"""
from pathlib import Path
import importlib.util
import numpy as np

REFERENCE = Path(__file__).resolve().parents[1]/"morphology_conductance/model.py"
spec=importlib.util.spec_from_file_location("demand_reference_conductance",REFERENCE)
reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
RULES=("exact","unit_broadcast","calibrated_broadcast")
NOMINAL=np.array([2.]*4+[.7]*4+[15.]*2+[20.,.2,20.,.2]+[4.]*2)

def teacher(seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,12345]))
    g=NOMINAL.copy()
    # Different local E/I ratios require the student to learn leaf response shapes.
    g[:4]=np.array([.25,8.,.25,8.])
    g[4:8]=np.array([8.,.25,8.,.25])
    return np.log(g)+rng.normal(0,.15,16)

def data(seed,kind,n,conflict,gate=10.):
    offsets={"train":101,"validation":102,"test":103,"calibration":104,"diagnostic":105}
    rng=np.random.default_rng(np.random.SeedSequence([seed,offsets[kind]]))
    context=rng.integers(2,size=n)
    latent=rng.uniform(-2.,2.,(n,2))
    other=rng.uniform(-2.,2.,(n,2))
    # A latent drives the selected subtree; the inactive one ranges from
    # independent distractor (conflict=0) to opposite latent (conflict=1).
    inactive=-conflict*latent+np.sqrt(1-conflict**2)*other
    xx=np.empty((n,4))
    for p in range(2): xx[:,2*p:2*p+2]=np.where((context==p)[:,None],latent,inactive)
    x=np.column_stack([np.exp(xx),np.exp(-xx),gate*(context==1),gate*(context==0)])
    y=reference.forward(teacher(seed)[None],x,reference.GROUPINGS[[0]])['output'][0]
    return x,y,context

def forward(theta,x):
    groups=np.tile(reference.GROUPINGS[0],(len(theta),1,1))
    return reference.forward(theta,x,groups)

def gradients(theta,x,y,variance,profiles,rules):
    groups=np.tile(reference.GROUPINGS[0],(len(theta),1,1))
    rule_ids=np.array([0 if r=='exact' else 1 for r in rules])
    pp=profiles.copy()
    for i,r in enumerate(rules):
        if r=='unit_broadcast': pp[i]=1.
    return reference.gradients(theta,x,y,groups,variance,pp,rule_ids)

def eligibility(theta,x):
    """Local derivative before multiplying six non-root path factors."""
    s=forward(theta,x); v,d,g=s['voltage'],s['denominator'],s['conductance']
    e=np.zeros((len(theta),len(x),16))
    e[:,:,:4]=x[None,:,:4]*(1-v[:,:,:4])/d[:,:,:4]*g[:,None,:4]
    e[:,:,4:8]=-x[None,:,4:8]*v[:,:,:4]/d[:,:,:4]*g[:,None,4:8]
    e[:,:,8:10]=-x[None,:,8:10]*v[:,:,4:6]/d[:,:,4:6]*g[:,None,8:10]
    for p in range(2):
        e[:,:,10+2*p:12+2*p]=(v[:,:,2*p:2*p+2]-v[:,:,4+p,None])/d[:,:,4+p,None]*g[:,None,10+2*p:12+2*p]
    e[:,:,14:16]=(v[:,:,4:6]-v[:,:,6,None])/d[:,:,6,None]*g[:,None,14:16]
    return e

PARAM_UNIT=np.array([0,1,2,3,0,1,2,3,4,5,4,4,5,5,6,6])

def metrics(theta,x,y,variance,profiles):
    state=forward(theta,x); q=state['path'][:,:,:6]
    error=(state['output']-y[None])/variance
    e=eligibility(theta,x)
    rows=[]
    for i in range(len(theta)):
        record={}
        for name,f in [('path',q[i]),('credit',q[i]*error[i,:,None])]:
            energy=float(np.sum(f*f)); eig=np.linalg.eigvalsh(f.T@f)
            record[name+'_rank_one_capture']=float(eig[-1]/max(energy,1e-30))
            record[name+'_effective_rank']=float(energy**2/max(np.sum(eig*eig),1e-30))
            for label,p in [('unit',np.ones(6)),('calibrated',profiles[i])]:
                record[name+'_'+label+'_oracle_capture']=float(np.sum((f@p)**2)/max(float(p@p)*energy,1e-30))
        # Current eligibilities weight six compartment path coordinates.
        weights=np.stack([np.sum(e[i,:,PARAM_UNIT==j]**2,axis=0) for j in range(6)],axis=1)
        den=float(np.sum(weights*q[i]**2))
        p=profiles[i]
        amp=np.sum(weights*q[i]*p,axis=1)/np.maximum(np.sum(weights*p*p,axis=1),1e-30)
        record['eligibility_calibrated_oracle_capture']=float(1-np.sum(weights*(q[i]-amp[:,None]*p)**2)/max(den,1e-30))
        ew=weights*error[i,:,None]**2
        amp=np.sum(ew*q[i]*p,axis=1)/np.maximum(np.sum(ew*p*p,axis=1),1e-30)
        record['loss_eligibility_calibrated_oracle_capture']=float(1-np.sum(ew*(q[i]-amp[:,None]*p)**2)/max(float(np.sum(ew*q[i]**2)),1e-30))
        rows.append(record)
    return rows
