"""Two E and two I channels per leaf, in the same directed conductance equation.

Four leaves, two parents, one soma. Leaf current is E/(1+E+I), parents
are sum(g_child*v_child)/(1+sum(g_child)+g_I*context), and soma is the same
balance without input inhibition. All 24 conductances are trainable.
Both subtrees receive identical positive and negative feature channels;
the task changes teacher E/I tuning, never the student's model or inputs.
"""
from pathlib import Path
import numpy as np
import model as first_model
REFERENCE=first_model.REFERENCE
RULES=first_model.RULES+("ancestry_three_oracle",)
NOMINAL=np.array([1.]*16+[15.]*2+[20.,.2,20.,.2]+[4.]*2)
PARAM_UNIT=np.array([j for j in range(4) for _ in range(4)]+[4,5,4,4,5,5,6,6])

def teacher(seed,opposed=True):
 rng=np.random.default_rng(np.random.SeedSequence([seed,12345]));g=NOMINAL.copy();local=np.array([[8.,.25,.25,8.]]*4)
 if opposed: local[2:]=[.25,8.,8.,.25]
 g[:16]=local.ravel();return np.log(g)+rng.normal(0,.15,24)

def data(seed,kind,n,conflict,gate=10.):
 offsets={'train':101,'validation':102,'test':103,'calibration':104,'diagnostic':105};rng=np.random.default_rng(np.random.SeedSequence([seed,offsets[kind]]));context=rng.integers(2,size=n);latent=rng.uniform(-2.,2.,(n,2));xx=np.tile(latent,(1,2));x=np.column_stack([np.exp(xx),np.exp(-xx),gate*(context==1),gate*(context==0)]);y=forward(teacher(seed,bool(conflict))[None],x)['output'][0];return x,y,context

def forward(theta,x):
 g=np.exp(theta);n=len(g);v=np.zeros((n,len(x),7));d=np.ones_like(v);gg=g[:,:16].reshape(n,4,4);xp=x[None,:,:4];xm=x[None,:,4:8];E=gg[:,None,:,0]*xp+gg[:,None,:,1]*xm;I=gg[:,None,:,2]*xp+gg[:,None,:,3]*xm;d[:,:,:4]=1+E+I;v[:,:,:4]=E/d[:,:,:4]
 for p in range(2):
  coupl=g[:,18+2*p:20+2*p];d[:,:,4+p]=1+coupl.sum(1)[:,None]+g[:,16+p,None]*x[None,:,8+p];v[:,:,4+p]=(v[:,:,2*p:2*p+2]*coupl[:,None,:]).sum(2)/d[:,:,4+p]
 d[:,:,6]=(1+g[:,22:24].sum(1))[:,None];v[:,:,6]=(v[:,:,4:6]*g[:,None,22:24]).sum(2)/d[:,:,6];q=np.ones_like(v);q[:,:,4:6]=g[:,None,22:24]/d[:,:,6,None]
 for p in range(2): q[:,:,2*p:2*p+2]=q[:,:,4+p,None]*g[:,None,18+2*p:20+2*p]/d[:,:,4+p,None]
 return dict(output=v[:,:,6],voltage=v,denominator=d,path=q,conductance=g)

def eligibility(theta,x):
 s=forward(theta,x);v,d,g=s['voltage'],s['denominator'],s['conductance'];n=len(theta);e=np.zeros((n,len(x),24));gg=g[:,:16].reshape(n,4,4);xp=x[None,:,:4];xm=x[None,:,4:8];ee=np.stack([(1-v[:,:,:4])*xp,(1-v[:,:,:4])*xm,-v[:,:,:4]*xp,-v[:,:,:4]*xm],axis=3)*gg[:,None]/d[:,:,:4,None];e[:,:,:16]=ee.reshape(n,len(x),16);e[:,:,16:18]=-x[None,:,8:10]*v[:,:,4:6]/d[:,:,4:6]*g[:,None,16:18]
 for p in range(2): e[:,:,18+2*p:20+2*p]=(v[:,:,2*p:2*p+2]-v[:,:,4+p,None])/d[:,:,4+p,None]*g[:,None,18+2*p:20+2*p]
 e[:,:,22:24]=(v[:,:,4:6]-v[:,:,6,None])/d[:,:,6,None]*g[:,None,22:24];return e

def ancestry_dictionary(profile):
 """Three fixed disjoint patterns: each distal subtree, and both parents."""
 D=np.zeros((6,3));D[:2,0]=profile[:2];D[2:4,1]=profile[2:4];D[4:,2]=profile[4:];return D

def project_ancestry(q,profile):
 D=ancestry_dictionary(profile);return q@D@np.diag(1/np.sum(D*D,axis=0))@D.T

def gradients(theta,x,y,variance,profiles,rules):
 s=forward(theta,x);q=s['path'].copy()
 for i,r in enumerate(rules):
  if r=='unit_broadcast': q[i,:,:6]=1.
  elif r=='calibrated_broadcast': q[i,:,:6]=profiles[i]
  elif r=='ancestry_three_oracle': q[i,:,:6]=project_ancestry(q[i,:,:6],profiles[i])
  else: assert r=='exact'
 e=eligibility(theta,x);gradient=np.mean((s['output']-y[None])[:,:,None]/variance*e*q[:,:,PARAM_UNIT],axis=1);return gradient,s

def metrics(theta,x,y,variance,profiles):
    state=forward(theta,x); q=state['path'][:,:,:6]
    error=(state['output']-y[None])/variance
    e=eligibility(theta,x)
    rows=[]
    for i in range(len(theta)):
        record={}
        for name,f in [('path',q[i]),('credit',q[i]*error[i,:,None])]:
            energy=float(np.sum(f*f)); eig=np.linalg.eigvalsh(f.T@f)
            projected=project_ancestry(f,profiles[i]);record[name+'_ancestry_three_oracle_capture']=float(np.sum(projected*projected)/max(energy,1e-30))
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
