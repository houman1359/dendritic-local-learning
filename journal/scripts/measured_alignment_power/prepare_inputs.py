#!/usr/bin/env python3
"""Archive only the public inputs needed to reproduce the alignment sensitivity test."""
from __future__ import annotations
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
J=Path(__file__).resolve().parents[2]
OUT=J/'source_data/measured_alignment_power'
EXTRACTS=J.parents[1]/'dendritic-credit-routing/results/microns_functional_partner_responses'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def partial(y,x,c):
 d=np.column_stack([np.ones(len(x)),*[rankdata(c[:,k]) for k in range(c.shape[1])]])
 xx=rankdata(x);yy=rankdata(y)
 xx-=d@np.linalg.lstsq(d,xx,rcond=None)[0];yy-=d@np.linalg.lstsq(d,yy,rcond=None)[0]
 return float(np.dot(xx,yy)/np.linalg.norm(xx)/np.linalg.norm(yy))
def main():
 inputs=OUT/'inputs';inputs.mkdir(exist_ok=True);manifest=[];scanrows=[];unitrows=[];checks=[]
 def copy(src,dst,definition):
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
  manifest.append(dict(source=str(src.resolve()),source_sha256=sha(src),released=str(dst.relative_to(OUT)),released_sha256=sha(dst),transformation='byte-identical copy',definition=definition))
 all_segments=pd.read_csv(J/'source_data/figure3/segment_metrics.csv',dtype={'root_id':'int64'})
 closure=[]
 for sp in sorted(EXTRACTS.glob('target*_automatic_conservative/functional_topology/summary.json')):
  raw=sp.parent.parent;ss=json.loads(sp.read_text());name=raw.name;dst=inputs/name;dst.mkdir(exist_ok=True)
  for fname,definition in [('functional_contact_pairs.csv','All unordered pairs of mapped presynaptic partners, with measured tuning similarity, ancestry and geometric covariates.'),('functional_contacts.csv','Ordered mapped partners, contact geometry and measured split-half Spearman reliability.'),('summary.json','Original per-scan functional-topology summary.')]:copy(sp.parent/fname,dst/fname,definition)
  contacts=pd.read_csv(sp.parent/'functional_contacts.csv',dtype={'pre_pt_root_id':'int64','nucleus_id':'int64'});pairs=pd.read_csv(sp.parent/'functional_contact_pairs.csv')
  psrc=raw/'microns_dandi_trial_protocol.npz';q=np.load(psrc,allow_pickle=False)
  msrc=raw/'microns_dandi_trial_unit_mapping.csv';m=pd.read_csv(msrc,dtype={'post_pt_root_id':'int64'})
  tsrc=raw/'microns_dandi_trial_table.csv';trial=pd.read_csv(tsrc)
  byroot={int(r):i for i,r in enumerate(q['unit_root_ids'])};cols=np.array([byroot[int(r)] for r in contacts.pre_pt_root_id],int)
  responses=q['responses'][:,cols];ids=q['stimulus_ids'];hashes=trial.condition_hash.astype(str).to_numpy();unique,cnt=np.unique(ids,return_counts=True);rep=unique[cnt>=2]
  order=sorted(set(hashes));assert np.array_equal(ids,np.array([order.index(v) for v in hashes]))
  assert np.isfinite(responses).all();assert len(responses)==len(trial)==464
  means=np.stack([responses[ids==v].mean(0) for v in rep]);left=[];right=[]
  for v in rep:
   ix=np.flatnonzero(ids==v);left.append(responses[ix[::2]].mean(0));right.append(responses[ix[1::2]].mean(0))
  left=np.array(left);right=np.array(right)
  reliab=np.array([spearmanr(left[:,k],right[:,k]).statistic for k in range(len(cols))])
  paircorr=np.corrcoef(means.T)[pairs.left.to_numpy(int),pairs.right.to_numpy(int)]
  rr=partial(paircorr,pairs.shared_path_fraction.to_numpy(),pairs[['euclidean_distance_um','path_depth_difference_um']].to_numpy())
  checks.append(dict(scan=name,max_reliability_difference=float(np.max(np.abs(reliab-contacts.repeat_reliability))),max_similarity_difference=float(np.max(np.abs(paircorr-pairs.functional_similarity))),partial_rank_recomputed=rr,partial_rank_original=ss['tests']['shared_path_partial_euclidean_and_depth']['partial_rank_r']))
  npz=dst/'observed_partner_responses.npz';np.savez_compressed(npz,responses=responses,stimulus_ids=ids,condition_hashes=np.asarray(hashes,dtype='U'),partner_root_ids=contacts.pre_pt_root_id.to_numpy(np.int64),half_left=left,half_right=right,repeated_stimulus_ids=rep)
  manifest.append(dict(source=[str(psrc.resolve()),str(msrc.resolve()),str(tsrc.resolve())],source_sha256=[sha(psrc),sha(msrc),sha(tsrc)],released=str(npz.relative_to(OUT)),released_sha256=sha(npz),transformation='Select only mapped presynaptic response columns by exact int64 root ID; retain all464trials, original condition identity and exact odd/even repeat means; omit task features, target trace and unrelated metadata.',definition='Public interval-averaged fluorescence responses, without additional deconvolution or reliability correction.'))
  obsmeta=[]
  for k,idx in enumerate(cols):
   row=m.iloc[int(idx)]
   assert int(row.post_pt_root_id)==int(contacts.pre_pt_root_id.iloc[k])
   rec=f"ses{int(row.session)}_scan{int(row.scan_idx)}_field{int(row.field)}_unit{int(row.unit_id)}"
   obsmeta.append(dict(partner_index=k,pre_pt_root_id=int(row.post_pt_root_id),recording_id=rec,session=int(row.session),scan_idx=int(row.scan_idx),field=int(row.field),unit_id=int(row.unit_id),dandi_asset_id=row.dandi_asset_id,dandi_path=row.dandi_path))
   unitrows.append(dict(scan=name,target_root_id=int(ss['target_root_id']),partner_index=k,pre_pt_root_id=int(row.post_pt_root_id),recording_id=rec,repeat_reliability=float(reliab[k]),segment_id=int(contacts.segment_id.iloc[k]),path_um=float(contacts.path_um.iloc[k])))
  pd.DataFrame(obsmeta).to_csv(dst/'recording_identity.csv',index=False)
  rp=dst/'recording_identity.csv';manifest.append(dict(source=str(msrc.resolve()),source_sha256=sha(msrc),released=str(rp.relative_to(OUT)),released_sha256=sha(rp),transformation='Select public identifiers for mapped partner columns.',definition='Observation identity for preserving duplicated optical recordings across target extracts.'))
  seg=all_segments[all_segments.root_id.eq(int(ss['target_root_id']))].copy();par=dict(zip(seg.segment_id.astype(int),seg.parent_segment_id.astype(int)));keep=set()
  for v in contacts.segment_id:
   v=int(v)
   while v>=0 and v not in keep:keep.add(v);v=par.get(v,-1)
  closure.append(seg[seg.segment_id.isin(keep)][['root_id','segment_id','parent_segment_id','path_length_um','edge_length_um']])
  scanrows.append(dict(scan=name,target_root_id=int(ss['target_root_id']),target_nucleus_id=int(ss['target_nucleus_id']),session=ss['session'],scan_idx=ss['scan_idx'],n_partners=len(contacts),n_pairs=len(pairs),n_trials=len(ids),n_repeated_stimuli=len(rep),repeats_two=int(sum(cnt==2)),repeats_ten=int(sum(cnt==10)),n_negative_reliabilities=int(sum(reliab<0)),observed_partial_rank=rr))
 pd.concat(closure).drop_duplicates(['root_id','segment_id']).sort_values(['root_id','segment_id']).to_csv(inputs/'ancestral_segment_closure.csv',index=False)
 p=inputs/'ancestral_segment_closure.csv';src=J/'source_data/figure3/segment_metrics.csv';manifest.append(dict(source=str(src.resolve()),source_sha256=sha(src),released=str(p.relative_to(OUT)),released_sha256=sha(p),transformation='Union of ancestor chains of mapped partner dominant-contact segments; select parent/length columns only.',definition='Minimal exact tree geometry required to reconstruct all cross-scan shared paths.'))
 pd.DataFrame(scanrows).to_csv(inputs/'scan_index.csv',index=False);pd.DataFrame(unitrows).to_csv(inputs/'partner_index.csv',index=False)
 copy(J/'code/task_derived/analyze_functional_topology.py',inputs/'original_functional_topology_analysis.py','Original statistic, repeat reliability and pair construction definitions; archived for audit, not modified.')
 copy(J/'scripts/aggregate_all_scan_functional_topology.py',inputs/'original_target_aggregation.py','Original scan-to-target mean and seven-target inferential procedure.')
 for p in [inputs/'scan_index.csv',inputs/'partner_index.csv']:
  manifest.append(dict(source='derived from the individually hashed inputs above',released=str(p.relative_to(OUT)),released_sha256=sha(p),transformation='Deterministic index only; no simulated outcomes.',definition='Portable dataset index.'))
 (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 a=pd.DataFrame(unitrows)
 report=dict(status='PASS',n_scans=len(scanrows),n_targets=a.target_root_id.nunique(),n_partner_observations=len(a),n_unique_partners=a.pre_pt_root_id.nunique(),n_roots_shared_across_targets=int((a.groupby('pre_pt_root_id').target_root_id.nunique()>1).sum()),n_target_partner_pairs_repeated_across_scans=int((a.groupby(['target_root_id','pre_pt_root_id']).size()>1).sum()),n_negative_repeat_reliability=int((a.repeat_reliability<0).sum()),checks=checks)
 assert len(scanrows)==13 and report['n_targets']==7
 assert max(x['max_reliability_difference'] for x in checks)<1e-12
 assert max(x['max_similarity_difference'] for x in checks)<1e-12
 assert max(abs(x['partial_rank_recomputed']-x['partial_rank_original']) for x in checks)<1e-12
 (OUT/'input_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
