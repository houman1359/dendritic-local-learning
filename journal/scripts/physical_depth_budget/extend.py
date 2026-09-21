#!/usr/bin/env python3
"""Frozen600-epoch extension of60 historical physical-depth conditions.

Original final files contain weights only, so all fits restart from their
original configuration and seed. The clean historical runtime is read-only.
A passive observer writes trajectories and resumable snapshots and evaluates
best-validation weights at180epochs with model/RNG restoration afterward.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent
JOURNAL=HERE.parents[1]
OUT=JOURNAL/'source_data/physical_depth_budget'
RUNTIME=Path('/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/h4-a99c3a7')
COMMIT='a99c3a777f99913e13dfe673a3f3a28bfe3566af'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def runtime_check():
    assert subprocess.check_output(['git','-C',str(RUNTIME),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(RUNTIME),'diff','HEAD','--','src','configs'],text=True)


def prepare():
    runtime_check();data=pd.read_csv(OUT/'original_epoch_diagnostics.csv')
    selected=data[data.depth.eq(3)| (data.depth.eq(1)&data.arm.eq('exact_autograd_bp_recipe'))].sort_values(['arm','depth','seed']).reset_index(drop=True)
    assert len(selected)==60
    configs=OUT/'extension_configs';configs.mkdir(exist_ok=True)
    records=[]
    for index,row in selected.iterrows():
        path=Path(row.config_path);assert sha(path)==row.config_sha256
        cfg=yaml.safe_load(path.read_text());original=copy.deepcopy(cfg)
        target=OUT/'extension_runs'/f'condition_{index:02d}'
        cfg['training']['main']['common']['epochs']=600
        cfg['outputs']['results_dir']=str(target)
        cfg['outputs']['run_name']=f'credit_depth_budget_{index:02d}'
        destination=configs/f'condition_{index:02d}.yaml'
        destination.write_text(yaml.safe_dump(cfg,sort_keys=False))
        records.append(dict(index=int(index),arm=row.arm,depth=int(row.depth),seed=int(row.seed),
            original_config=str(path),original_config_sha256=sha(path),config=str(destination),config_sha256=sha(destination),
            original_training_summary=row.training_summary_path,original_training_summary_sha256=row.training_summary_sha256,
            original_final_metrics=row.final_metrics_path,original_final_model=row.final_model_path,
            results_dir=str(target),original_early_stopping=original['training']['main']['common']['early_stopping'],
            original_patience=original['training']['main']['common']['patience']))
    inputs=[HERE/'extend.py',HERE/'worker.sh',HERE/'audit_existing.py',OUT/'original_epoch_diagnostics.csv',
            RUNTIME/'src/dendritic_modeling/training/strategies/standard.py',
            RUNTIME/'src/dendritic_modeling/training/strategies/local_learning.py',
            RUNTIME/'src/dendritic_modeling/scripts/training/train_experiments.py']
    protocol=dict(observer_version=3,observer_fix='Serialize wrapped optimizer and resolve timestamped pipeline output directory; benchmark training retained. No accepted600-epoch outcomes preceded repair',hardware_assignment='Before extension outcomes: seeds10200..10203use GPU,max4concurrent; seeds10204..10209use CPU,max16concurrent. All six arms within a seed use the same device class.',status='Frozen before extension outcomes; existing-seed budget sensitivity, not fresh replication',
        conditions=records,runtime=str(RUNTIME),runtime_commit=COMMIT,
        changes='Only maximum epochs180to600 and output paths/names; same seeds, tasks, model, optimizer, rates, early-stopping patience30, validation-based final selection',
        checkpoint_audit='Inspected historical final_model.pt: OrderedDict of35model state entries; no optimizer, RNG, minibatch state. Runtime checkpoint saver also stores model state only. Restart required.',
        stopping='Original validation-loss improvement rule and patience30 preserved; max600; a falling curve at cap is not called converged',
        reference='Five D3credit/optimizer arms and exact-autograd BP D1reference,10paired originalseeds each',
        primary_outcome='Held-out test accuracy at validation-selected endpoint within600epochs;180epoch window reported side by side',
        contrasts=['D3exact_BP minus D1exact_BP','D3path_LocalCA minus D3shared_LocalCA','D3exact_BP minus D3broadcast_BP','D3broadcast_BP minus D3broadcast_LocalCA_recipe'],
        uncertainty='Descriptive paired whole-seed bootstrap95%intervals; no new multiplicity-adjusted confirmatory claim',
        replay='Compare first180(or original early-stopped length) validation trajectory with archived source. Report full discrepancies; clean immutable source/GPU changes may prevent bitwise identity. No outcome-dependent replacement.',
        instrumentation='Observer only: saves progress, best-state180evaluation with restoration of model state/training mode and Python/NumPy/Torch CPU/CUDA RNG; restart snapshots at180,300,600 or original stopping',
        source_sha256={str(p):sha(p) for p in inputs})
    path=OUT/'extension_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:dump(path,protocol);dump(OUT/'extension_freeze.json',dict(protocol_sha256=sha(path),created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print('Frozen',len(records),'conditions')


def optimizer_snapshot(optimizer):
    wrappers=[]
    while not hasattr(optimizer, 'state_dict'):
        wrappers.append(dict(type=type(optimizer).__name__, weight_decay=getattr(optimizer,'weight_decay',None), weight_boosting=getattr(optimizer,'weight_boosting',None)))
        optimizer=optimizer.optimizer
    return dict(state=optimizer.state_dict(),wrappers=wrappers)


def rng_state(torch):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(torch,state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def run(index,benchmark=False):
    runtime_check();protocol=json.loads((OUT/'extension_protocol.json').read_text())
    for p,digest in protocol['source_sha256'].items():assert sha(p)==digest,p
    record=protocol['conditions'][index];assert sha(record['config'])==record['config_sha256']
    destination=Path(record['results_dir'])
    config=Path(record['config'])
    if benchmark:
        destination=OUT/'benchmark'/f'condition_{index:02d}'
        cfg=yaml.safe_load(config.read_text());cfg['training']['main']['common']['epochs']=20
        cfg['outputs']['results_dir']=str(destination);cfg['outputs']['run_name']='depth_budget_benchmark'
        config=OUT/'benchmark_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    if (destination/'extension_audit.json').exists():raise FileExistsError('Completed condition immutable')
    destination.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(RUNTIME/'src'))
    import torch
    torch.set_num_threads(1)
    from dendritic_modeling.training.strategies.standard import Trainer
    from dendritic_modeling.scripts.training import train_experiments
    assert Path(train_experiments.__file__).resolve().is_relative_to(RUNTIME)
    original_epoch=Trainer._run_epoch
    progress=[];tick=time.perf_counter()
    def observed_epoch(self,*args,**kwargs):
        result=original_epoch(self,*args,**kwargs)
        model=kwargs.get('model',args[0] if args else None)
        epoch=int(self.epoch_counter)
        progress.append(dict(epoch=epoch,train_loss=float(result[0][-1]),valid_loss=float(result[1][-1]),
            best_loss=float(self.best_loss),best_epoch=int(self.best_epoch),patience_counter=int(self.patience_counter),
            elapsed_seconds=time.perf_counter()-tick))
        dump(destination/'extension_progress.json',progress)
        stopping=self.early_stopping and self.patience_counter>=self.patience
        if epoch in (180,300,600) or stopping or (benchmark and epoch==20):
            rng=rng_state(torch)
            current=copy.deepcopy(self._unwrap_model(model).state_dict())
            snapshot=dict(epoch=epoch,current_model=current,best_model=copy.deepcopy(self.best_state_dict),
                optimizer=optimizer_snapshot(self.optimizer),rng=rng,train_losses=result[0],valid_losses=result[1],
                best_loss=self.best_loss,best_epoch=self.best_epoch,patience_counter=self.patience_counter,
                source_commit=COMMIT,protocol_sha256=sha(OUT/'extension_protocol.json'))
            torch.save(snapshot,destination/f'extension_state_{epoch}.pt')
            if epoch==180:
                was_training=model.training
                try:
                    self._unwrap_model(model).load_state_dict(self.best_state_dict)
                    self.analysis_manager.run_analysis(filename='budget180_best',training=False)
                finally:
                    self._unwrap_model(model).load_state_dict(current);model.train(was_training);restore_rng(torch,rng)
        return result
    Trainer._run_epoch=observed_epoch
    try:train_experiments.main(str(config))
    finally:Trainer._run_epoch=original_epoch
    summaries=list(destination.glob('*/training_summary.json'))
    assert len(summaries)==1,summaries
    model_results=summaries[0].parent
    summary=json.loads(summaries[0].read_text())
    original=json.loads(Path(record['original_training_summary']).read_text())
    n=min(len(summary['valid_losses']),len(original['valid_losses']),180)
    difference=np.array(summary['valid_losses'][:n])-np.array(original['valid_losses'][:n])
    dump(destination/'extension_audit.json',dict(status='complete',index=index,arm=record['arm'],depth=record['depth'],seed=record['seed'],
        benchmark=benchmark,model_results_dir=str(model_results),epochs=len(summary['valid_losses']),max_epochs=20 if benchmark else 600,
        runtime_commit=COMMIT,config_sha256=sha(config),protocol_sha256=sha(OUT/'extension_protocol.json'),
        first_window_epochs=n,archived_validation_max_abs_difference=float(abs(difference).max()),
        archived_validation_rms_difference=float(np.sqrt(np.mean(difference*difference))),
        source_concordance='Report differences explicitly; not assumed bitwise replication',
        elapsed_seconds=time.perf_counter()-tick,torch=torch.__version__,python=sys.version,
        device='cuda' if torch.cuda.is_available() else'cpu',slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        output_sha256={str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file() and p.suffix in ('.pt','.json') and p.name!='extension_audit.json'}))
    print(json.dumps(json.loads((destination/'extension_audit.json').read_text())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','benchmark']);parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:run(args.index,args.action=='benchmark')
