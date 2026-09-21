#!/usr/bin/env python3
"""Replay verified conductance fits in a relocated or different library environment.

The frozen runners are unchanged. Their original environment equality guard is
not invoked: original identities are verified through the release hash chain,
and original and actual library versions are recorded separately. Only an
explicit excluded smoke budget can shorten the scientific update budget.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

HERE=Path(__file__).resolve().parent
DATA_PREFIX=Path('source_data/conductance_credit_demand')
# Immutable anchor for the882-file scientific handoff; new portable files have
# a separate inventory and do not rewrite this original experimental record.
SCIENCE_INVENTORY_SHA256='8f2eda1084dfdddc566bb38fef50c9d10f2811f6302993ce3576a112ffe1059f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def safe_child(root,relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe relative frozen source path')
    child=(root/relative).resolve()
    if not child.is_relative_to(root.resolve()):raise ValueError('Frozen source escapes its root')
    return child

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

def input_context(study_root,journal_root,family):
    journal=Path(journal_root).resolve();given=Path(study_root).resolve()
    base=given.parent if family=='opponent' and given.name=='opponent' else given
    study=base/'opponent' if family=='opponent' else base
    helper_path=journal/'code/release_noise/release_hashes.py'
    if not helper_path.is_file():raise FileNotFoundError('The restored reviewer package must include code/release_noise/release_hashes.py')
    helper=load_module('_conductance_portable_release_hashes',helper_path)
    verified=[]
    def verify(path,original,canonical):
        verdict=helper.verify_released_file(path,original,journal_root=journal)
        if not verdict['verified']:raise ValueError(f'Unverified frozen input {canonical}: {verdict["reason"]}')
        verified.append(dict(path=str(canonical),original_sha256=original,released_sha256=verdict['released_sha256'],reason=verdict['reason']))
        return verdict
    inventory=base/'science_handoff_inventory_20260906.tsv'
    verify(inventory,SCIENCE_INVENTORY_SHA256,DATA_PREFIX/'science_handoff_inventory_20260906.tsv')
    with inventory.open(newline='') as stream:rows=list(csv.DictReader(stream,delimiter='\t'))
    indexed={}
    for row in rows:
        key=Path(row['path']).as_posix()
        if key in indexed:raise ValueError('Duplicate scientific inventory path')
        indexed[key]=row['sha256']
    prefix=DATA_PREFIX/'opponent' if family=='opponent' else DATA_PREFIX
    def verify_inventory(relative,expected=None):
        relative=Path(relative);key=relative.as_posix()
        if key not in indexed and expected is None:raise ValueError(f'Frozen inventory omits {key}')
        # The original freeze also pins a shared reference outside this study's
        # own-file inventory. Its expectation comes from that verified record.
        original=indexed.get(key,expected)
        if expected is not None and original!=expected:raise ValueError(f'Frozen digest records disagree for {key}')
        path=safe_child(base,relative.relative_to(DATA_PREFIX)) if relative.is_relative_to(DATA_PREFIX) else safe_child(journal,relative)
        verify(path,original,relative);return path
    freeze_path=verify_inventory(prefix/'development_freeze.json')
    freeze=json.loads(freeze_path.read_text())
    protocol_path=verify_inventory(prefix/'protocol.json',freeze['protocol_sha256'])
    cfg=json.loads(protocol_path.read_text())
    for relative,digest in freeze['source_sha256'].items():verify_inventory(relative,digest)
    selection_path=verify_inventory(prefix/'selection_freeze.json')
    selection=json.loads(selection_path.read_text())
    original_freeze=indexed[(prefix/'development_freeze.json').as_posix()]
    if selection['development_freeze_sha256']!=original_freeze:raise ValueError('Selection does not identify the original development freeze')
    extension_name='extend_opponent.py' if family=='opponent' else 'extend.py'
    verify_inventory(Path('scripts/conductance_credit_demand')/extension_name)
    if family=='opponent':
        ext=json.loads(verify_inventory(prefix/'extension_source_freeze.json').read_text())
        verify_inventory(ext['source'],ext['sha256'])
    if cfg['steps']!=4096 or cfg['checkpoints']!=[0,64,256,1024,2048,4096]:
        raise ValueError('Unsupported original budget/checkpoint contract')
    return dict(journal=journal,base=base,study=study,cfg=cfg,freeze=freeze,
                selection=selection,verified=verified,verify_inventory=verify_inventory)

def load_frozen(context,family):
    code=context['journal']/'scripts/conductance_credit_demand'
    # These generic original module names are isolated in the launcher process.
    # Refuse an unrelated preimported module rather than silently reusing it.
    for name in ['model','opponent_model']:
        if name in sys.modules:
            actual=Path(getattr(sys.modules[name],'__file__','')).resolve()
            if actual!=code/f'{name}.py':raise ValueError(f'Conflicting preimported frozen module: {name}')
    sys.path.insert(0,str(code))
    name='run_opponent' if family=='opponent' else 'run'
    frozen=load_module('_conductance_verified_fit_runner',code/f'{name}.py')
    expected_model=code/('opponent_model.py' if family=='opponent' else 'model.py')
    if Path(frozen.model.__file__).resolve()!=expected_model:raise ValueError('Fit imported an unverified model')
    if Path(frozen.model.REFERENCE).resolve()!=context['journal']/'scripts/morphology_conductance/model.py':
        raise ValueError('Fit imported an unexpected conductance reference')
    if context['cfg'] != frozen.default_config():
        raise ValueError('Released protocol differs from the verified original scientific defaults')
    # Reading the already verified selection from a relocated study is an I/O
    # change only. Calling run_task avoids the historical full-record guard.
    frozen.OUT=context['study']
    return frozen

def execution_plan(context,phase,seed,smoke_steps=None):
    cfg=copy.deepcopy(context['cfg']);original_phase='development' if phase=='development' else 'fresh'
    if seed not in cfg[original_phase+'_seeds']:raise ValueError('Seed is not part of the declared cohort')
    if smoke_steps is not None and not 1<=smoke_steps<=64:raise ValueError('Excluded smoke budget must be between1 and64 updates')
    budget=16384 if phase=='extension' else 4096
    if phase=='extension':cfg['checkpoints']=cfg['checkpoints']+[8192,12288,16384]
    steps=budget if smoke_steps is None else smoke_steps
    tasks=cfg['tasks']
    if original_phase=='fresh':tasks=[t for t in tasks if t['name'] in context['selection']['confirmatory_tasks']]
    if not tasks:raise ValueError('The frozen selection has no confirmatory tasks')
    return cfg,original_phase,steps,tasks,budget

def execute(args):
    context=input_context(args.study_root,args.journal_root,args.family)
    cfg,original_phase,steps,tasks,full_budget=execution_plan(context,args.phase,args.seed,args.excluded_smoke_steps)
    frozen=load_frozen(context,args.family)
    import numpy as np
    import pandas as pd
    actual_versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    original_versions={key:context['freeze'][key] for key in ['python','numpy','pandas']}
    identity=dict(family=args.family,requested_phase=args.phase,original_fit_phase=original_phase,seed=args.seed,
                  original_environment=original_versions,actual_environment=actual_versions,
                  original_environment_equality_asserted=False,verified_inputs=context['verified'],
                  science_inventory_original_sha256=SCIENCE_INVENTORY_SHA256,
                  portable_launcher_sha256=sha(Path(__file__)))
    if args.verify_only:
        print(json.dumps(dict(status='passed',mode='verify_only',**identity),indent=2));return dict(status='passed',**identity)
    if args.output_root is None:raise ValueError('A new --output-root is required for fitting')
    out=Path(args.output_root).resolve()
    if out.exists():raise FileExistsError('Portable execution refuses an existing output directory')
    if out.is_relative_to(context['base']):raise ValueError('Portable output must be outside the restored study')
    out.mkdir(parents=True)
    mode='excluded_smoke' if args.excluded_smoke_steps is not None else 'portable_existing_seed_replay'
    declared=['relocated study input and new output directories','actual library environment recorded independently of original']
    if args.phase=='extension':declared.append('full16384-update replay from original initialization with the predeclared extended checkpoint cadence')
    if args.excluded_smoke_steps is not None:declared.append('explicit excluded smoke update budget')
    audit=dict(status='running',mode=mode,excluded_from_scientific_results=mode=='excluded_smoke',steps=steps,
               original_complete_budget=full_budget,declared_changes=declared,**identity)
    dump(out/'portable_audit.json',audit);dump(out/'execution_protocol.json',cfg)
    start=time.perf_counter();curves=[];diagnostics=[];endpoints=[];files={}
    try:
        for task in tasks:
            c,d,e,metadata,arrays=frozen.run_task(args.seed,task,original_phase,cfg,steps=steps)
            curves+=c;diagnostics+=d;endpoints+=e
            path=out/f'{task["name"]}_states.npz';np.savez_compressed(path,**arrays);files[path.name]=sha(path)
            path=out/f'{task["name"]}_metadata.json';dump(path,metadata);files[path.name]=sha(path)
        for kind,rows in [('curves',curves),('diagnostics',diagnostics),('endpoints',endpoints)]:
            path=out/f'{kind}.csv';pd.DataFrame(rows).to_csv(path,index=False);files[path.name]=sha(path)
        audit.update(status='passed',n_fits=len(endpoints),elapsed_seconds=time.perf_counter()-start,output_sha256=files,
                     reproduction_scope='Verified scientific source/protocol and original seed; numerical library changes can affect floating-point trajectories. Extended replay restarts the same stream from initialization, rather than claiming to import historical optimizer state.')
    except Exception as error:
        audit.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-start)
        dump(out/'portable_audit.json',audit);raise
    dump(out/'portable_audit.json',audit);print(json.dumps(audit,indent=2));return audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True,help='Restored conductance_credit_demand root, or its opponent subdirectory for that family')
    p.add_argument('--journal-root',type=Path,default=HERE.parents[1],help='Restored journal containing verified scripts and release_noise helper')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--family',choices=['first','opponent'],required=True)
    p.add_argument('--phase',choices=['development','fresh','extension'],default='fresh')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--excluded-smoke-steps',type=int,help='Explicit1–64-update canary, always excluded from scientific results')
    p.add_argument('--verify-only',action='store_true')
    execute(p.parse_args())
