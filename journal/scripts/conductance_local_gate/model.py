"""Independent credit adapter for the released seven-compartment circuit.

Local rules never evaluate exact path factors in their gradient computation.
Exact/projection comparators are explicitly separate branches.
"""
import numpy as np

NOMINAL=np.array([1.]*16+[15.]*2+[20.,.2,20.,.2]+[4.]*2)

def teacher(seed,opposed=True):
    rng=np.random.default_rng(np.random.SeedSequence([seed,12345]));g=NOMINAL.copy();local=np.array([[8.,.25,.25,8.]]*4)
    if opposed:local[2:]=[.25,8.,8.,.25]
    g[:16]=local.ravel();return np.log(g)+rng.normal(0,.15,24)

def data(seed,kind,n,conflict,gate=10.):
    offsets={'train':101,'validation':102,'test':103,'calibration':104,'diagnostic':105}
    rng=np.random.default_rng(np.random.SeedSequence([seed,offsets[kind]]));context=rng.integers(2,size=n);latent=rng.uniform(-2.,2.,(n,2));xx=np.tile(latent,(1,2))
    x=np.column_stack([np.exp(xx),np.exp(-xx),gate*(context==1),gate*(context==0)])
    y=forward(teacher(seed,bool(conflict))[None],x)['output'][0]
    return x,y,context

PARAM_UNIT=np.array([j for j in range(4) for _ in range(4)]+[4,5,4,4,5,5,6,6])
LOCAL_RULES={'hard_distal_unit_proximal','swapped_distal_unit_proximal','hard_distal_and_proximal','shunt_proportional_unit_proximal'}
RULES=('exact','unit_broadcast','calibrated_broadcast','ancestry_three_oracle',
       'hard_distal_unit_proximal','swapped_distal_unit_proximal','hard_distal_and_proximal',
       'ancestry_two_leaf_oracle_unit_proximal','shunt_proportional_unit_proximal')

def forward(theta,x):
    g=np.exp(theta);n=len(g);v=np.zeros((n,len(x),7));d=np.ones_like(v)
    gg=g[:,:16].reshape(n,4,4);xp=x[None,:,:4];xm=x[None,:,4:8]
    E=gg[:,None,:,0]*xp+gg[:,None,:,1]*xm
    I=gg[:,None,:,2]*xp+gg[:,None,:,3]*xm
    d[:,:,:4]=1+E+I;v[:,:,:4]=E/d[:,:,:4]
    for p in range(2):
        co=g[:,18+2*p:20+2*p]
        d[:,:,4+p]=1+co.sum(1)[:,None]+g[:,16+p,None]*x[None,:,8+p]
        v[:,:,4+p]=(v[:,:,2*p:2*p+2]*co[:,None,:]).sum(2)/d[:,:,4+p]
    d[:,:,6]=(1+g[:,22:24].sum(1))[:,None]
    v[:,:,6]=(v[:,:,4:6]*g[:,None,22:24]).sum(2)/d[:,:,6]
    return {'voltage':v,'denominator':d,'conductance':g,'output':v[:,:,6]}

def exact_path(state):
    g,d,v=state['conductance'],state['denominator'],state['voltage'];q=np.ones_like(v)
    q[:,:,4:6]=g[:,None,22:24]/d[:,:,6,None]
    for p in range(2):
        q[:,:,2*p:2*p+2]=q[:,:,4+p,None]*g[:,None,18+2*p:20+2*p]/d[:,:,4+p,None]
    return q

def eligibility(state,x):
    v,d,g=state['voltage'],state['denominator'],state['conductance'];n=len(g)
    e=np.zeros((n,len(x),24));gg=g[:,:16].reshape(n,4,4);xp=x[None,:,:4];xm=x[None,:,4:8]
    ee=np.stack([(1-v[:,:,:4])*xp,(1-v[:,:,:4])*xm,-v[:,:,:4]*xp,-v[:,:,:4]*xm],axis=3)*gg[:,None]/d[:,:,:4,None]
    e[:,:,:16]=ee.reshape(n,len(x),16)
    e[:,:,16:18]=-x[None,:,8:10]*v[:,:,4:6]/d[:,:,4:6]*g[:,None,16:18]
    for p in range(2):
        e[:,:,18+2*p:20+2*p]=(v[:,:,2*p:2*p+2]-v[:,:,4+p,None])/d[:,:,4+p,None]*g[:,None,18+2*p:20+2*p]
    e[:,:,22:24]=(v[:,:,4:6]-v[:,:,6,None])/d[:,:,6,None]*g[:,None,22:24]
    return e

def subset(state,i):
    return {k:v[i:i+1] for k,v in state.items()}

def delivered(state,x,profiles,rules):
    q=np.ones_like(state['voltage'])
    for i,r in enumerate(rules):
        if r=='unit_broadcast':continue
        if r=='calibrated_broadcast':q[i,:,:6]=profiles[i];continue
        if r in LOCAL_RULES:
            active=(x[:,8:10]==0).astype(float)
            if r=='swapped_distal_unit_proximal':active=1-active
            if r=='shunt_proportional_unit_proximal':active=1/(1+state['conductance'][i,16:18][None,:]*x[:,8:10])
            q[i,:,:4]=np.repeat(active,2,axis=1)
            if r=='hard_distal_and_proximal':q[i,:,4:6]=active
            continue
        current=exact_path(subset(state,i))[0]
        if r=='exact':q[i]=current;continue
        assert r in {'ancestry_three_oracle','ancestry_two_leaf_oracle_unit_proximal'}
        groups=[(0,1),(2,3),(4,5)] if r=='ancestry_three_oracle' else [(0,1),(2,3)]
        for group in groups:
            ix=np.array(group);p=profiles[i,ix];amplitude=current[:,ix]@p/(p@p)
            q[i][:,ix]=amplitude[:,None]*p
    return q

def gradients(theta,x,y,variance,profiles,rules):
    state=forward(theta,x);q=delivered(state,x,profiles,rules);e=eligibility(state,x)
    gradient=np.mean((state['output']-y[None])[:,:,None]/variance*e*q[:,:,PARAM_UNIT],axis=1)
    return gradient,state
