"""Portable rerun of a frozen physical-depth condition using exact a99c3a7 sources.

The original extend.py is preserved. This launcher verifies its helper source,
the historical runtime and released configuration, then applies only declared
output relocation (or an explicitly labeled short smoke budget).
"""
from __future__ import annotations
import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

from release_hashes import digest, verify_released_file

COMMIT='a99c3a777f99913e13dfe673a3f3a28bfe3566af'


def expected_for(protocol, suffix):
    values=[value for key,value in protocol['source_sha256'].items() if key.endswith(suffix)]
    if len(values)!=1:raise ValueError(f'Frozen source identity is not unique: {suffix}')
    return values[0]


def verify(path, original):
    result=verify_released_file(path,original)
    if not result['verified']:raise ValueError(f'Unverified frozen input {path}: {result["reason"]}')
    return result


def load_verified_inputs(source_root, journal, runtime, index):
    freeze=json.loads((source_root/'extension_freeze.json').read_text())
    protocol_path=source_root/'extension_protocol.json'
    verify(protocol_path,freeze['protocol_sha256'])
    protocol=json.loads(protocol_path.read_text())
    if protocol['runtime_commit']!=COMMIT:raise ValueError('Unexpected historical runtime commit')
    metadata=json.loads((runtime/'RUNTIME_PROVENANCE.json').read_text())
    if metadata['commit']!=COMMIT:raise ValueError('Runtime export does not identify the frozen commit')
    with (runtime/'RUNTIME_ORIGINS.tsv').open(newline='') as handle:
        sources=list(csv.DictReader(handle,delimiter='\t'))
    for row in sources:
        if row['commit']!=COMMIT:raise ValueError('Mixed runtime commits')
        verify(runtime/row['path'],row['original_sha256'])
    for relative in ['src/dendritic_modeling/training/strategies/standard.py',
                     'src/dendritic_modeling/training/strategies/local_learning.py',
                     'src/dendritic_modeling/scripts/training/train_experiments.py']:
        verify(runtime/relative,expected_for(protocol,relative))
    observer=journal/'scripts/physical_depth_budget/extend.py'
    verify(observer,expected_for(protocol,'scripts/physical_depth_budget/extend.py'))
    condition=next(row for row in protocol['conditions'] if row['index']==index)
    candidates=[source_root/'configurations'/f'condition_{index:02d}_extension.yaml',
                source_root/'extension_configs'/f'condition_{index:02d}.yaml']
    config_path=next((p for p in candidates if p.is_file()),None)
    if config_path is None:raise FileNotFoundError('Restore the canonical extension configuration first')
    verify(config_path,condition['config_sha256'])
    cfg=yaml.safe_load(config_path.read_text())
    if cfg['experiment']['seed']!=condition['seed']:raise ValueError('Configuration seed differs')
    if cfg['training']['main']['common']['epochs']!=600:raise ValueError('Frozen extension budget differs')
    if cfg['training']['main']['common']['patience']!=condition['original_patience']:
        raise ValueError('Early-stopping patience differs')
    return protocol,condition,cfg,observer,dict(runtime_commit=COMMIT,runtime_files_verified=len(sources),
        protocol_original_sha256=freeze['protocol_sha256'],protocol_released_sha256=digest(protocol_path),
        config_original_sha256=condition['config_sha256'],config_released_sha256=digest(config_path),
        frozen_observer_original_sha256=expected_for(protocol,'scripts/physical_depth_budget/extend.py'))


def run(args):
    source_root=args.source_root.resolve();journal=args.journal_root.resolve();runtime=args.runtime_root.resolve()
    protocol,condition,cfg,observer,identity=load_verified_inputs(source_root,journal,runtime,args.condition)
    sys.path.insert(0,str(runtime/'src'))
    import torch
    torch.set_num_threads(1)
    from dendritic_modeling.training.strategies.standard import Trainer
    from dendritic_modeling.scripts.training import train_experiments
    if not Path(train_experiments.__file__).resolve().is_relative_to(runtime):
        raise ValueError('The imported trainer escaped the verified historical runtime')
    identity.update(torch=torch.__version__,numpy=np.__version__,python=sys.version,
                    condition=args.condition,arm=condition['arm'],seed=condition['seed'])
    if args.verify_only:
        print(json.dumps(dict(status='passed',mode='verify_only',**identity),indent=2));return
    if args.output_root is None:raise ValueError('Set a new --output-root for rerunning training')
    output=args.output_root.resolve()
    if output.exists():raise FileExistsError('Portable reruns require a new output directory')
    output.mkdir(parents=True)
    budget=args.smoke_epochs or 600
    cfg['outputs']['results_dir']=str(output/'training')
    cfg['outputs']['run_name']=f'portable_depth_{args.condition:02d}'
    if args.smoke_epochs:cfg['training']['main']['common']['epochs']=budget
    config=output/'execution_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    spec=importlib.util.spec_from_file_location('_frozen_depth_observer',observer)
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    progress=[];original_epoch=Trainer._run_epoch;started=time.perf_counter()
    def observe(self,*positional,**keywords):
        result=original_epoch(self,*positional,**keywords)
        model=keywords.get('model',positional[0] if positional else None);epoch=int(self.epoch_counter)
        progress.append(dict(epoch=epoch,train_loss=float(result[0][-1]),valid_loss=float(result[1][-1]),
            best_loss=float(self.best_loss),best_epoch=int(self.best_epoch),patience_counter=int(self.patience_counter)))
        frozen.dump(output/'progress.json',progress)
        stopping=self.early_stopping and self.patience_counter>=self.patience
        if epoch in (180,300,600) or stopping or epoch==budget:
            rng=frozen.rng_state(torch);current=copy.deepcopy(self._unwrap_model(model).state_dict())
            torch.save(dict(epoch=epoch,current_model=current,best_model=copy.deepcopy(self.best_state_dict),
                optimizer=frozen.optimizer_snapshot(self.optimizer),rng=rng,train_losses=result[0],valid_losses=result[1],
                best_loss=self.best_loss,best_epoch=self.best_epoch,patience_counter=self.patience_counter,
                source_commit=COMMIT,protocol_sha256=identity['protocol_original_sha256']),output/f'state_{epoch}.pt')
            if epoch==180:
                was_training=model.training
                try:
                    self._unwrap_model(model).load_state_dict(self.best_state_dict)
                    self.analysis_manager.run_analysis(filename='budget180_best',training=False)
                finally:
                    self._unwrap_model(model).load_state_dict(current);model.train(was_training);frozen.restore_rng(torch,rng)
        return result
    Trainer._run_epoch=observe
    try:train_experiments.main(str(config))
    finally:Trainer._run_epoch=original_epoch
    summaries=list((output/'training').glob('*/training_summary.json'))
    if len(summaries)!=1:raise ValueError('Unexpected training-output layout')
    summary=json.loads(summaries[0].read_text());metrics=json.loads((summaries[0].parent/'performance/final.json').read_text())
    report=dict(status='passed',mode='excluded_smoke' if args.smoke_epochs else 'portable_existing_seed_rerun',
        **identity,epochs=len(summary['valid_losses']),max_epochs=budget,execution_config_sha256=digest(config),
        declared_changes=['output directory and run name']+(['explicit smoke budget'] if args.smoke_epochs else []),
        elapsed_seconds=time.perf_counter()-started,endpoint_metrics=metrics,
        selection='Original minimum-validation-loss rule and early-stopping patience; no test-based selection',
        reproduction_scope='Exact selected historical source files; device/library versions can prevent bitwise trajectory identity')
    frozen.dump(output/'portable_audit.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True,help='Restored physical_depth_budget/canonical Source Data')
    parser.add_argument('--journal-root',type=Path,required=True)
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--condition',type=int,choices=range(60),required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--smoke-epochs',type=int,choices=range(1,21))
    run(parser.parse_args())
