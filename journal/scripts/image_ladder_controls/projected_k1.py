#!/usr/bin/env python3
"""Separately frozen exact-amplitude K1 addendum to the five-rule MNIST study."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd
import yaml
import run as base
OUT=base.OUT/'projected_k1'
ARM='projected_k1'
STAGES={'development':'projected_k1_development','fresh':'projected_k1_fresh','canary':'projected_k1_canary'}

def project_common(transported):
    distal,proximal,soma=transported
    if any(x is None for x in transported):raise ValueError('Missing exact field')
    b,n=soma.shape
    assert distal.shape==(b,n*9) and proximal.shape==(b,n*3)
    coefficient=(distal.reshape(b,n,9).sum(-1)+proximal.reshape(b,n,3).sum(-1))/12
    return [coefficient.unsqueeze(-1).expand(-1,-1,9).reshape_as(distal),coefficient.unsqueeze(-1).expand(-1,-1,3).reshape_as(proximal),soma]

def install_hooks():
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_transport_mixin import LocalLearningPathTransportBroadcastMixin
    original=LocalLearningPathTransportBroadcastMixin._precompute_path_transport_errors
    def common(self,*args,**kwargs):return project_common(original(self,*args,**kwargs))
    LocalLearningPathTransportBroadcastMixin._precompute_path_transport_errors=common
    return {'transport':original}

def prepare():
    base.runtime_check()
    if (OUT/'protocol.json').exists():raise FileExistsError('Addendum already frozen')
    base.MODES[ARM]='path_transport'
    dev=base.make_conditions(STAGES['development'],[(a,ARM,m,s) for a in base.ARCHITECTURES for m in base.RATES for s in base.DEV_SEEDS])
    canary=base.make_conditions(STAGES['canary'],[(a,ARM,1.,50999) for a in base.ARCHITECTURES])
    for rec in canary:
        path=Path(rec['config']);cfg=yaml.safe_load(path.read_text());cfg['training']['main']['common']['epochs']=3;path.write_text(yaml.safe_dump(cfg,sort_keys=False));rec['config_sha256']=base.sha(path)
    base.dump(base.OUT/f"{STAGES['canary']}_conditions.json",canary)
    protocol=dict(status='Separate addendum frozen before any fresh outcomes or projectedK1 trainingoutcomes',
        parent_protocol_sha256=base.sha(base.OUT/'protocol.json'),reason='Isolate spatialresolution fromexample-dependent commonamplitude: K3 andK1 both project exactactivationfields',
        arm=ARM,rule='One exact-fieldmean coefficient perneuron across12nonsomatic activationerrors; somaerror remains exact; all localeligibilities unchanged',
        new_primary_spatial_contrast='subtree_k3 minus projected_k1 at selectedrates andcommonoriginalrate',
        development_count=18,development_seeds=base.DEV_SEEDS,fresh_seeds=base.FRESH_SEEDS,rate_multipliers=base.RATES,
        epochs=180,selection='Minimum mean validation-selected loss over3developmentseeds; tie1then.3then3; testoutcomes unused',
        fresh_design='10freshpairedseeds perarchitecture atselectedrate plus originalcommon1ifdifferent; identicaldata/model/minibatchseeds tofive-armcohort',
        exclusions='Two3epochseed50999canaries excluded; nooptionalstopping',
        frozen_parent_adapter_sha256=base.sha(base.HERE/'run.py'),source_sha256={str(base.HERE/name):base.sha(base.HERE/name) for name in ('projected_k1.py','projected_k1_worker.sh','advance_addendum.py')})
    base.dump(OUT/'protocol.json',protocol);base.dump(OUT/'freeze.json',dict(protocol_sha256=base.sha(OUT/'protocol.json'),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('Frozen18development plus2excludedcanary fits')

def run(phase,index):
    protocol=json.loads((OUT/'protocol.json').read_text());assert base.sha(OUT/'protocol.json')==json.loads((OUT/'freeze.json').read_text())['protocol_sha256']
    assert base.sha(base.HERE/'run.py')==protocol['frozen_parent_adapter_sha256']
    for p,digest in protocol['source_sha256'].items():assert base.sha(p)==digest,p
    original=base.install_hooks
    def hooks(arm):return install_hooks() if arm==ARM else original(arm)
    base.install_hooks=hooks
    stage=STAGES[phase];base.run(stage,index)
    rec=json.loads((base.OUT/f'{stage}_conditions.json').read_text())[index]
    audit=Path(rec['results_dir'])/'run_audit.json';value=json.loads(audit.read_text())
    value.update(addendum_protocol_sha256=base.sha(OUT/'protocol.json'),addendum_adapter_sha256=base.sha(base.HERE/'projected_k1.py'))
    base.dump(audit,value)

def select():
    records=json.loads((base.OUT/f"{STAGES['development']}_conditions.json").read_text());rows=[]
    for rec in records:
        root=Path(rec['results_dir']);audit=json.loads((root/'run_audit.json').read_text());assert audit['epochs']==180 and audit['status']=='complete'
        progress=json.loads((root/'progress.json').read_text());rows.append(dict(architecture=rec['architecture'],arm=ARM,multiplier=rec['multiplier'],seed=rec['seed'],validation_loss=min(x['valid_loss'] for x in progress)))
    data=pd.DataFrame(rows);data.to_csv(OUT/'development_selection_rows.csv',index=False)
    means=data.groupby(['architecture','arm','multiplier'],as_index=False).validation_loss.mean();means.to_csv(OUT/'development_selection_means.csv',index=False)
    chosen=[];design=[]
    for architecture in base.ARCHITECTURES:
        g=means[means.architecture.eq(architecture)].copy();g['tie']=g.multiplier.map({1.:0,.3:1,3.:2});r=g.sort_values(['validation_loss','tie']).iloc[0];m=float(r.multiplier)
        chosen.append(dict(architecture=architecture,arm=ARM,multiplier=m,development_validation_loss=float(r.validation_loss)))
        design.extend((architecture,ARM,rate,seed) for rate in sorted({1.,m}) for seed in base.FRESH_SEEDS)
    path=OUT/'selection.json'
    if path.exists():raise FileExistsError('Addendum selection alreadyfrozen')
    base.dump(path,dict(selected=chosen,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),protocol_sha256=base.sha(OUT/'protocol.json'),criterion='Onlydevelopmentvalidationloss'))
    base.MODES[ARM]='path_transport';base.make_conditions(STAGES['fresh'],design);print('Frozen',len(design),'projectedK1freshfits')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','select']);p.add_argument('--phase',choices=list(STAGES),default='development');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='select':select()
    else:run(a.phase,a.index)
