#!/usr/bin/env python3
"""Frozen MNIST delivery dictionary, learning-rate and decoder-floor controls.

Only isolated process-local hooks extend the clean historical implementation.
K3 averages *activation-space* exact error within each proximal+children group.
Local activation derivatives and all other eligibilities are unchanged.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.metadata, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
HERE=Path(__file__).resolve().parent
JOURNAL=HERE.parents[1]
PROJECT=JOURNAL.parents[2]
OUT=JOURNAL/'source_data/image_ladder_controls'
RAW=Path('/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260820/sweep_runs/image_ladder_controls_20260906')
RUNTIME=Path('/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/mnist-6c1aaa')
COMMIT='6c1aaa25abd056c417842e1c46378b65d036f6a7'
OLD=RAW.parent/'mnist_feedback_ladder'
ARCHITECTURES=('shunting','additive')
ARMS=('strict_scalar','neuron_shared','subtree_k3','exact_path','decoder_only')
RATES=(0.3,1.0,3.0)
DEV_SEEDS=(50200,50201,50202)
FRESH_SEEDS=tuple(range(50300,50310))
MODES={'strict_scalar':'scalar','neuron_shared':'per_soma_shared','subtree_k3':'path_transport','exact_path':'path_transport','decoder_only':'per_soma_shared'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def runtime_check():
    assert subprocess.check_output(['git','-C',str(RUNTIME),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(RUNTIME),'diff','HEAD','--','src','configs'],text=True)
def source_config(architecture):
    stamp='20260825173517' if architecture=='shunting' else '20260825173519'
    return OLD/f'journal_mnist_feedback_ladder_exact_path_{architecture}_15seed_{stamp}'/'configs/unified_config_0.yaml'
def make_conditions(stage, designs):
    records=[]
    for i,(architecture,arm,multiplier,seed) in enumerate(designs):
        cfg=yaml.safe_load(source_config(architecture).read_text())
        cfg['experiment']['seed']=seed
        cfg['data']['base_dir']=str(PROJECT/'data')
        cfg['training']['main']['learning_strategy_config']['error_broadcast_mode']=MODES[arm]
        rates=cfg['training']['main']['common']['param_groups']
        for key in ('lr','topk_lr','blocklinear_lr','reactivation_lr','decoder_lr'): rates[key]*=multiplier
        cfg['outputs']['results_dir']=str(RAW/stage/f'condition_{i:03d}')
        cfg['outputs']['run_name']=f'mnist_{stage}_{i:03d}'
        cfg['_sweep_config_id']=f'condition_{i:03d}'
        if stage=='canary': cfg['training']['main']['common']['epochs']=3
        path=OUT/'configs'/stage/f'condition_{i:03d}.yaml'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        records.append(dict(index=i,stage=stage,architecture=architecture,arm=arm,multiplier=multiplier,seed=seed,
            config=str(path),config_sha256=sha(path),results_dir=cfg['outputs']['results_dir']))
    dump(OUT/f'{stage}_conditions.json',records)
    return records

def prepare():
    runtime_check()
    if (OUT/'protocol.json').exists(): raise FileExistsError('Existing protocol is immutable')
    source_hashes={str(p):sha(p) for p in (HERE/'run.py',HERE/'worker.sh')}
    runtime_files={str(p.relative_to(RUNTIME)):sha(p) for p in sorted((RUNTIME/'src').rglob('*.py'))}
    raw=PROJECT/'data/mnist/MNIST/raw'
    data_hashes={str(p.relative_to(PROJECT/'data')):sha(p) for p in sorted(raw.iterdir()) if p.is_file()}
    dev=make_conditions('development',[(a,r,m,s) for a in ARCHITECTURES for r in ARMS for m in RATES for s in DEV_SEEDS])
    make_conditions('canary',[(a,r,1.0,50999) for a in ARCHITECTURES for r in ARMS])
    protocol=dict(status='Frozen before development/fresh outcomes; canaries excluded',version=1,
        runtime_commit=COMMIT,runtime=str(RUNTIME),runtime_sha256=runtime_files,source_sha256=source_hashes,
        original_config_sha256={a:sha(source_config(a)) for a in ARCHITECTURES},mnist_cache_sha256=data_hashes,
        architectures=ARCHITECTURES,arms=ARMS,rate_multipliers=RATES,development_seeds=DEV_SEEDS,fresh_seeds=FRESH_SEEDS,
        development_count=len(dev),epochs=180,batch_size=256,early_stopping=False,selection_checkpoint='minimum validation cross-entropy within180epochs',
        rate_selection='For each architecture and arm: minimum mean validation-selected loss over3developmentseeds; tie prefers multiplier1,then0.3,then3. Test outcomes never used to choose.',
        fresh_design='10new paired seeds per architecture/arm at selected rate; also original common multiplier1when selected differs; reuse identical selected/common fit when multiplier1chosen.',
        primary_contrasts=['neuron_shared minus strict_scalar accuracy','subtree_k3 minus neuron_shared accuracy','exact_path minus subtree_k3 accuracy','neuron_shared minus decoder_only accuracy'],
        secondary_metrics=['test cross-entropy','minimum validation loss','best epoch','full validation/train curves'],
        inference='Paired whole-seed bootstrap95%intervals; seed is inferential unit. All outcomes including dev and any failure retained. Fixed180epoch comparison, not proof of convergence.',
        delivery_coordinate='Activation output at each compartment. Local activation derivative stays in eligibility. Soma coordinate remains exact except strictscalar arm.',
        subtree_k3='At every batch exact transported activation errors are computed, then averaged within each proximal compartment and its3distalchildren. Three nonoverlapping indicator columns per neuron. This is oracle-informed delivery; coefficient acquisition is not biologically local. Soma error unchanged.',
        decoder_only='Same initialized core and contact mask, normal180epoch full minibatch stream and local linear decoder updates; skip only core/gate gradient writeback. Validate unchanged core parameter hashes. No feature caching or altered minibatches.',
        match='Within seed all arms and rates share data/model/topology/loader seeds and originalmodelparameterization; no input normalization, augmentation or reduceddata. Architecture-specific initialization retained.',
        historical_capture='Archived atlas uses pre-reactivation voltage error; new delivered-space capture uses activation error and is reported distinctly.',
        exclusions='Ten3epochseed50999 canaries excluded from all selection and scientific summaries. No optional stopping.')
    dump(OUT/'protocol.json',protocol)
    dump(OUT/'freeze.json',dict(protocol_sha256=sha(OUT/'protocol.json'),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('Frozen90development fits,10excludedcanaries',flush=True)

def project_subtrees(transported):
    """Euclidean projection in output-activation coordinates, soma-major layout."""
    distal,proximal,soma=transported
    if any(x is None for x in transported): raise ValueError('Missing exact error field')
    b,n=soma.shape
    assert proximal.shape==(b,n*3) and distal.shape==(b,n*9)
    d=distal.reshape(b,n,3,3);p=proximal.reshape(b,n,3)
    coefficients=(p+d.sum(-1))/4
    return [coefficients.unsqueeze(-1).expand(-1,-1,-1,3).reshape_as(distal),coefficients.reshape_as(proximal),soma]

def tensor_hash(state,prefix=None):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        if prefix is not None and not name.startswith(prefix):continue
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def install_hooks(arm):
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_transport_mixin import LocalLearningPathTransportBroadcastMixin
    from dendritic_modeling.training.strategies.local_learning_parts.local_learning_rule_mixin import LocalLearningRuleMixin
    originals={}
    if arm=='subtree_k3':
        original=LocalLearningPathTransportBroadcastMixin._precompute_path_transport_errors
        originals['transport']=original
        def routed(self,*args,**kwargs): return project_subtrees(original(self,*args,**kwargs))
        LocalLearningPathTransportBroadcastMixin._precompute_path_transport_errors=routed
    if arm=='decoder_only':
        originals['core']=LocalLearningRuleMixin._apply_local_rule_gradients
        def skip_core(self,*args,**kwargs): return None
        LocalLearningRuleMixin._apply_local_rule_gradients=skip_core
    return originals

def run(stage,index):
    runtime_check()
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(OUT/'protocol.json')==json.loads((OUT/'freeze.json').read_text())['protocol_sha256']
    for path,digest in protocol['source_sha256'].items(): assert sha(path)==digest,path
    record=json.loads((OUT/f'{stage}_conditions.json').read_text())[index]
    assert sha(record['config'])==record['config_sha256']
    destination=Path(record['results_dir'])
    if (destination/'run_audit.json').exists(): raise FileExistsError('Completed outcome is immutable')
    destination.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(RUNTIME/'src'))
    import torch
    torch.set_num_threads(1)
    from dendritic_modeling.training.strategies.standard import Trainer
    from dendritic_modeling.scripts.training import train_experiments
    assert Path(train_experiments.__file__).resolve().is_relative_to(RUNTIME)
    originals=install_hooks(record['arm'])
    original_epoch=Trainer._run_epoch
    progress=[];initial={};tick=time.perf_counter()
    def observed_epoch(self,*args,**kwargs):
        model=kwargs.get('model',args[0] if args else None)
        if not initial:
            state=model.state_dict()
            initial.update(model_sha256=tensor_hash(state),core_sha256=tensor_hash(state,'core_network.'),
                decoder_sha256=tensor_hash(state,'decoder_network.'),parameter_count=sum(p.numel() for p in model.parameters()))
            dump(destination/'initial_state_identity.json',initial)
            torch.save(state,destination/'initial_model.pt')
            optimizer=self.optimizer
            wrappers=[]
            while hasattr(optimizer,'optimizer'):
                wrappers.append(type(optimizer).__name__);optimizer=optimizer.optimizer
            dump(destination/'optimizer.json',dict(type=type(optimizer).__name__,wrappers=wrappers,
                groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]))
        result=original_epoch(self,*args,**kwargs)
        progress.append(dict(epoch=int(self.epoch_counter),train_loss=float(result[0][-1]),valid_loss=float(result[1][-1]),
            best_loss=float(self.best_loss),best_epoch=int(self.best_epoch),elapsed_seconds=time.perf_counter()-tick))
        dump(destination/'progress.json',progress)
        return result
    Trainer._run_epoch=observed_epoch
    try: train_experiments.main(record['config'])
    finally: Trainer._run_epoch=original_epoch
    summaries=list(destination.rglob('training_summary.json'));assert len(summaries)==1,summaries
    results=summaries[0].parent;summary=json.loads(summaries[0].read_text())
    state=torch.load(results/'final_model.pt',map_location='cpu',weights_only=False)
    finalcore=tensor_hash(state,'core_network.');finaldecoder=tensor_hash(state,'decoder_network.')
    if record['arm']=='decoder_only': assert initial['core_sha256']==finalcore,'Decoder floor changed core'
    assert initial['decoder_sha256']!=finaldecoder,'Decoder did not change'
    versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','pandas','omegaconf','PyYAML','wandb')}
    audit=dict(**record,status='complete',runtime_commit=COMMIT,protocol_sha256=sha(OUT/'protocol.json'),
        model_results_dir=str(results),epochs=len(summary['valid_losses']),initial=initial,final_core_sha256=finalcore,
        final_decoder_sha256=finaldecoder,elapsed_seconds=time.perf_counter()-tick,versions=versions,python=sys.version,
        device='cuda' if torch.cuda.is_available() else 'cpu',cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        output_sha256={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file() and p.suffix in ('.pt','.json') and p.name!='run_audit.json'})
    dump(destination/'run_audit.json',audit)
    print(json.dumps({k:audit[k] for k in ('stage','index','arm','status','elapsed_seconds','epochs')}),flush=True)

def select():
    records=json.loads((OUT/'development_conditions.json').read_text());rows=[]
    for rec in records:
        audit=json.loads((Path(rec['results_dir'])/'run_audit.json').read_text())
        assert audit['status']=='complete' and audit['epochs']==180
        progress=json.loads((Path(rec['results_dir'])/'progress.json').read_text())
        rows.append(dict(architecture=rec['architecture'],arm=rec['arm'],multiplier=rec['multiplier'],seed=rec['seed'],validation_loss=min(x['valid_loss'] for x in progress)))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'development_selection_rows.csv',index=False)
    means=frame.groupby(['architecture','arm','multiplier'],as_index=False).validation_loss.mean();means.to_csv(OUT/'development_selection_means.csv',index=False)
    selected=[];design=[]
    for a in ARCHITECTURES:
        for r in ARMS:
            group=means[means.architecture.eq(a)&means.arm.eq(r)].copy()
            group['tie']=group.multiplier.map({1.0:0,0.3:1,3.0:2})
            row=group.sort_values(['validation_loss','tie']).iloc[0];m=float(row.multiplier)
            selected.append(dict(architecture=a,arm=r,multiplier=m,development_validation_loss=float(row.validation_loss)))
            for mult in sorted(set([1.0,m])):
                for seed in FRESH_SEEDS: design.append((a,r,mult,seed))
    path=OUT/'selection.json'
    if path.exists(): raise FileExistsError('Selection already frozen')
    dump(path,dict(selected=selected,created_utc=pd.Timestamp.now(tz='UTC').isoformat(),criterion='Only development validation loss',protocol_sha256=sha(OUT/'protocol.json')))
    make_conditions('fresh',design)
    print('Frozen',len(design),'fresh fits')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','select']);p.add_argument('--stage',choices=['development','fresh','canary'],default='development');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='select':select()
    else:run(a.stage,a.index)
