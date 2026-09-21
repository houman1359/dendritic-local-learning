"""Response-level conditional sensitivity model; pairs are never simulated independently."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
@dataclass
class Scan:
 name:str
 target:int
 roots:np.ndarray
 stimuli:np.ndarray
 recordings:np.ndarray
 counts:np.ndarray
 r:np.ndarray
 h:float
 left:np.ndarray
 right:np.ndarray
 q:np.ndarray
 x:np.ndarray
 pair_x:np.ndarray
 controls:np.ndarray
@dataclass
class Dataset:
 scans:list[Scan]
 roots:np.ndarray
 hashes:list[str]
 recordings:list[str]
 kernel:np.ndarray
 factor:np.ndarray
 scan_target:np.ndarray
 targets:np.ndarray
 metadata:dict

def residual_design(x,controls):
 d=np.column_stack([np.ones(len(x)),*[rankdata(controls[:,k]) for k in range(controls.shape[1])]])
 u,s,_=np.linalg.svd(d,full_matrices=False);q=u[:,s>s[0]*1e-12]
 xx=rankdata(x);xx=xx-q@(q.T@xx)
 if np.linalg.norm(xx)<1e-12:raise ValueError('Ancestry has no residual variation after the declared controls.')
 return q,xx/np.linalg.norm(xx)

def load_dataset(root:Path)->Dataset:
 root=Path(root);index=pd.read_csv(root/'scan_index.csv',dtype={'target_root_id':'int64'})
 partners=pd.read_csv(root/'partner_index.csv',dtype={'pre_pt_root_id':'int64','target_root_id':'int64'})
 segments=pd.read_csv(root/'ancestral_segment_closure.csv',dtype={'root_id':'int64'})
 roots=np.sort(partners.pre_pt_root_id.unique());ri={int(r):k for k,r in enumerate(roots)}
 targets=np.sort(index.target_root_id.unique());ti={int(r):k for k,r in enumerate(targets)}
 covariance=np.zeros((len(roots),len(roots)));memberships=np.zeros(len(roots));kernel_audits=[]
 for target in targets:
  ct=partners[partners.target_root_id.eq(target)].drop_duplicates('pre_pt_root_id').sort_values('pre_pt_root_id')
  assert partners[partners.target_root_id.eq(target)].groupby('pre_pt_root_id').segment_id.nunique().max()==1
  ss=segments[segments.root_id.eq(target)];parents=dict(zip(ss.segment_id.astype(int),ss.parent_segment_id.astype(int)));length=dict(zip(ss.segment_id.astype(int),ss.path_length_um))
  chain={}
  for seg in ct.segment_id:
   chain[int(seg)]=[];v=int(seg)
   while v>=0:chain[int(seg)].append(v);v=parents.get(v,-1)
  ids=[int(x) for x in ct.segment_id];loc=np.array([ri[int(x)] for x in ct.pre_pt_root_id]);k=np.zeros((len(ids),len(ids)))
  for a,sa in enumerate(ids):
   for b,sb in enumerate(ids):
    common=next(v for v in chain[sb] if v in set(chain[sa]));denom=np.sqrt(length[sa]*length[sb]);k[a,b]=length[common]/denom if denom>0 else float(a==b)
  assert np.allclose(np.diag(k),1);minimum=float(np.linalg.eigvalsh(k).min());assert minimum>-1e-10
  covariance[np.ix_(loc,loc)]+=k;memberships[loc]+=1
  kernel_audits.append(dict(target_root_id=int(target),n_partners=len(ids),minimum_eigenvalue=minimum))
 covariance/=np.sqrt(memberships[:,None]*memberships[None,:]);ev,evec=np.linalg.eigh(covariance);assert ev.min()>-1e-10
 factor=evec*np.sqrt(np.maximum(ev,0))[None,:]
 hashset=set()
 for name in index.scan:
  obs=np.load(root/name/'observed_partner_responses.npz');mask=np.isin(obs['stimulus_ids'],obs['repeated_stimulus_ids']);hashset.update(obs['condition_hashes'][mask].tolist())
 hashes=sorted(hashset)
 hi={h:k for k,h in enumerate(hashes)};recordings=sorted(partners.recording_id.unique());oi={v:k for k,v in enumerate(recordings)}
 scans=[];recording_values={}
 for row in index.itertuples(index=False):
  src=root/row.scan;contacts=pd.read_csv(src/'functional_contacts.csv',dtype={'pre_pt_root_id':'int64'});pairs=pd.read_csv(src/'functional_contact_pairs.csv');obs=np.load(src/'observed_partner_responses.npz');identity=pd.read_csv(src/'recording_identity.csv')
  ids=obs['stimulus_ids'];hasharr=obs['condition_hashes'];unique,cnt=np.unique(ids,return_counts=True);repeat=unique[cnt>=2];counts=cnt[cnt>=2]
  keys=np.array([hi[str(hasharr[ids==v][0])] for v in repeat]);roots_idx=np.array([ri[int(v)] for v in contacts.pre_pt_root_id]);records_idx=np.array([oi[v] for v in identity.recording_id])
  raw=contacts.repeat_reliability.to_numpy();r=2*np.sin(np.pi*np.clip(raw,0,1)/6)
  # Equal odd/even half sizes for this dataset; pooled Gaussian moment calibration.
  assert np.all(counts%2==0);h=float(np.mean(2/counts))
  for n,rr in zip(records_idx,r):
   if n in recording_values:assert abs(rr-recording_values[n])<1e-12
   recording_values[n]=rr
  controls=pairs[['euclidean_distance_um','path_depth_difference_um']].to_numpy();x=pairs.shared_path_fraction.to_numpy();q,xnorm=residual_design(x,controls)
  # Verify own-scan Brownian kernel reconstruction against the released pair records.
  ct_by=partners[partners.target_root_id.eq(row.target_root_id)].drop_duplicates('pre_pt_root_id').set_index('pre_pt_root_id')
  for pp in pairs.itertuples(index=False):
   aa=contacts.iloc[int(pp.left)];bb=contacts.iloc[int(pp.right)]
   assert abs(pp.shared_path_um/max(min(aa.path_um,bb.path_um),1e-9)-pp.shared_path_fraction)<1e-12
  scans.append(Scan(row.scan,ti[int(row.target_root_id)],roots_idx,keys,records_idx,counts.astype(float),r,h,pairs.left.to_numpy(int),pairs.right.to_numpy(int),q,xnorm,x,controls))
 metadata=dict(n_scans=len(scans),n_targets=len(targets),n_unique_partners=len(roots),n_unique_recordings=len(recordings),n_unique_stimuli=len(hashes),global_kernel_min_eigenvalue=float(ev.min()),target_kernels=kernel_audits,global_kernel_definition='Sum the seven target-specific Brownian shared-path correlation kernels in global presynaptic-root coordinates; divide by square roots of diagonal membership counts. The two cross-target shared roots receive one consistent response, not two independent simulated neurons.',model_scope='Conditional Gaussian stimulus-tuning model on the observed 102 presynaptic roots. Does not establish sensitivity to missing partners, endogenous teaching, arbitrary effect shapes or biological between-target dependence beyond recorded shared roots.')
 return Dataset(scans,roots,hashes,recordings,covariance,factor,np.array([s.target for s in scans]),targets,metadata)

def similarities(y:np.ndarray,scan:Scan):
 """One Pearson correlation matrix from jointly generated partner responses per draw."""
 z=y-y.mean(axis=1,keepdims=True);norm=np.sqrt(np.sum(z*z,axis=1))
 return np.einsum('bti,btj->bij',z,z,optimize=True)/(norm[:,:,None]*norm[:,None,:])

def statistic(y:np.ndarray,scan:Scan):
 corr=similarities(y,scan);v=corr[:,scan.left,scan.right]
 ranked=rankdata(v,axis=1);res=ranked-(ranked@scan.q)@scan.q.T
 return (res@scan.x)/np.linalg.norm(res,axis=1)

def aggregate(scans:np.ndarray,scan_target:np.ndarray,n_targets:int):
 return np.stack([scans[...,scan_target==k].mean(axis=-1) for k in range(n_targets)],axis=-1)

def exact_signed_rank(values:np.ndarray):
 """Enumerate all 2^n sign assignments; supports average absolute ranks and exact zeros."""
 values=np.asarray(values,float);n=values.shape[-1];shape=values.shape[:-1];flat=values.reshape(-1,n);out=np.empty(len(flat))
 nonzero=(flat!=0).all(axis=1)
 if nonzero.any():
  x=flat[nonzero];ranks=rankdata(np.abs(x),axis=1);obs=np.minimum((ranks*(x>0)).sum(1),(ranks*(x<0)).sum(1))
  bits=np.array(list(itertools.product([0,1],repeat=n)),float)
  sums=ranks@bits.T;low=np.minimum(sums,ranks.sum(1)[:,None]-sums)
  out[nonzero]=np.mean(low<=obs[:,None]+1e-12,axis=1)
 for i in np.flatnonzero(~nonzero):
  x=flat[i][flat[i]!=0];out[i]=1. if len(x)==0 else float(exact_signed_rank(x[None,:])[0])
 return out.reshape(shape)

def exact_test_lattice(n=7):
 bits=np.array(list(itertools.product([0,1],repeat=n)),float);values=(2*bits-1)*np.arange(1,n+1)
 p=exact_signed_rank(values);return dict(n=n,n_sign_assignments=len(bits),minimum_two_sided_p=float(p.min()),null_rejection_fraction_at_005=float(np.mean(p<=.05)),rejecting_negative_rank_sums=sorted(set(int(x) for x in (((values<0)*np.arange(1,n+1)).sum(1))[p<=.05])),distinct_pvalues=sorted(set(p.tolist())))
