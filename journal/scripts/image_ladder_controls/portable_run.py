#!/usr/bin/env python3
"""Replay one frozen MNIST control from a restored reviewer package.

Accepts explicit installation/data/output locations. Scientific configuration
and official MNIST bytes are verified independently of relocatable I/O paths.
The historical runtime may supply an original-to-released hash inventory when
its text was sanitized by the release builder; original identities must match
the study's frozen runtime inventory in either case.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def scientific_signature(config):
    value=copy.deepcopy(config)
    value['data'].pop('base_dir',None);value.pop('outputs',None);value.pop('_sweep_config_id',None)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def condition_signature(stage,record,config_signature):
    """Bind adapter dispatch to the condition, independently of relocatable paths."""
    value=dict(stage=stage,index=int(record['index']),architecture=record['architecture'],
               arm=record['arm'],multiplier=float(record['multiplier']),seed=int(record['seed']),
               config_scientific_sha256=config_signature)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_runtime(root,protocol,release_inventory=None):
    released={}
    if release_inventory:
        import csv
        with release_inventory.open() as handle:
            released={row['path']:row for row in csv.DictReader(handle,delimiter='\t')}
    checked=0
    for name,original in protocol['runtime_sha256'].items():
        path=root/name
        if name in released:
            record=released[name]
            if record['original_sha256']!=original:raise ValueError(f'Runtime identity mismatch: {name}')
            expected=record['released_sha256']
        else:expected=original
        if not path.is_file() or sha(path)!=expected:raise ValueError(f'Runtime byte mismatch: {name}')
        checked+=1
    return checked

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True,help='Parent containing mnist/MNIST/raw official dataset files')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--runtime-release-inventory',type=Path,help='TSV columns path, original_sha256, released_sha256; required only for transformed runtime files')
    p.add_argument('--adapter-release-record',type=Path,help='JSON original_sha256/released_sha256 for the frozen adapter when its I/O paths were sanitized')
    p.add_argument('--stage',choices=['development','fresh','canary','projected_k1_development','projected_k1_fresh','projected_k1_canary'],default='fresh')
    p.add_argument('--projected-k1-release-record',type=Path,help='Original/released source record for the separately frozen K1 adapter')
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    study=args.study_root.resolve();runtime=args.runtime_root.resolve();dataset=args.dataset_root.resolve();out=args.output_root.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Replay output must be new or empty')
    protocol=json.loads((study/'protocol.json').read_text())
    inv=json.loads((study/'portable_scientific_inventory.json').read_text())
    # The publication allowlist contains the training/library dependency closure,
    # not unrelated projectdrivers that happened to share the historicalrepo.
    subset=dict(protocol,runtime_sha256=inv['runtime_required_sha256'])
    for name,digest in subset['runtime_sha256'].items():
        if protocol['runtime_sha256'].get(name)!=digest:raise ValueError('Allowlist changed historical identity')
    checked=validate_runtime(runtime,subset,args.runtime_release_inventory)
    for name,digest in protocol['mnist_cache_sha256'].items():
        path=dataset/name
        if not path.is_file() or sha(path)!=digest:raise ValueError(f'MNIST byte mismatch: {name}')
    record=json.loads((study/f'{args.stage}_conditions.json').read_text())[args.index]
    cfg_path=study/'configs'/args.stage/f'condition_{args.index:03d}.yaml'
    cfg=yaml.safe_load(cfg_path.read_text())
    inv=json.loads((study/'portable_scientific_inventory.json').read_text())
    key=f'{args.stage}/condition_{args.index:03d}'
    if scientific_signature(cfg)!=inv['config_scientific_sha256'][key]:raise ValueError('Scientific configuration was changed')
    if condition_signature(args.stage,record,scientific_signature(cfg))!=inv['condition_scientific_sha256'][key]:raise ValueError('Scientific condition or delivery rule was changed')
    sys.path.insert(0,str(runtime/'src'))
    # Import the unchanged, hash-frozen experimental adapter after runtime setup.
    import run as frozen
    adapter_expected=inv['adapter_sha256']
    if args.adapter_release_record:
        release=json.loads(args.adapter_release_record.read_text())
        if release['original_sha256']!=adapter_expected:raise ValueError('Adapter original identity mismatch')
        adapter_expected=release['released_sha256']
    if sha(HERE/'run.py')!=adapter_expected:raise ValueError('Frozen delivery adapter was changed')
    import torch
    torch.set_num_threads(1)
    from dendritic_modeling.scripts.training import train_experiments
    from dendritic_modeling.training.strategies.standard import Trainer
    assert Path(train_experiments.__file__).resolve().is_relative_to(runtime)
    if record['arm']=='projected_k1':
        expected=inv['projected_k1_adapter_sha256']
        if args.projected_k1_release_record:
            link=json.loads(args.projected_k1_release_record.read_text())
            if link['original_sha256']!=expected:raise ValueError('K1 adapter original identity mismatch')
            expected=link['released_sha256']
        if sha(HERE/'projected_k1.py')!=expected:raise ValueError('K1 adapter bytes changed')
        import projected_k1
        projected_k1.install_hooks()
    else:frozen.install_hooks(record['arm'])
    # These are the only changes made to the saved resolved configuration.
    cfg['data']['base_dir']=str(dataset)
    cfg['outputs']['results_dir']=str(out)
    cfg['outputs']['run_name']=f'mnist_replay_{args.stage}_{args.index:03d}'
    out.mkdir(parents=True,exist_ok=True)
    config=out/'relocated_config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    assert scientific_signature(cfg)==inv['config_scientific_sha256'][key]
    original=Trainer._run_epoch;initial={};progress=[];tick=time.perf_counter()
    def observe(self,*a,**kw):
        model=kw.get('model',a[0] if a else None)
        if not initial:
            state=model.state_dict()
            initial.update(model_sha256=frozen.tensor_hash(state),core_sha256=frozen.tensor_hash(state,'core_network.'),decoder_sha256=frozen.tensor_hash(state,'decoder_network.'))
            torch.save(state,out/'initial_model.pt')
        result=original(self,*a,**kw)
        progress.append(dict(epoch=int(self.epoch_counter),train_loss=float(result[0][-1]),valid_loss=float(result[1][-1]),best_loss=float(self.best_loss),best_epoch=int(self.best_epoch)))
        dump(out/'replay_progress.json',progress)
        return result
    Trainer._run_epoch=observe
    try:train_experiments.main(str(config))
    finally:Trainer._run_epoch=original
    results=list(out.rglob('training_summary.json'));assert len(results)==1
    final=torch.load(results[0].parent/'final_model.pt',map_location='cpu',weights_only=False)
    core=frozen.tensor_hash(final,'core_network.')
    if record['arm']=='decoder_only':assert core==initial['core_sha256']
    executed={}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename).resolve()
        if path.is_relative_to(runtime) and path.suffix=='.py':
            name=str(path.relative_to(runtime))
            if name not in subset['runtime_sha256']:raise ValueError(f'Executed source omitted from allowlist: {name}')
            executed[name]=sha(path)
    dump(out/'replay_audit.json',dict(executed_runtime_files=executed,stage=args.stage,index=args.index,arm=record['arm'],status='complete',initial=initial,
        final_core_sha256=core,epochs=len(progress),runtime_files_verified=checked,runtime_commit=protocol['runtime_commit'],
        configuration_scientific_sha256=scientific_signature(cfg),adapter_sha256=sha(HERE/'run.py'),
        elapsed_seconds=time.perf_counter()-tick,torch=torch.__version__,python=sys.version,
        protocol='Original epochs, data, seeds, rule, optimizer and validation selection retained; only I/O paths relocated'))
    print(out/'replay_audit.json',flush=True)
if __name__=='__main__':main()
