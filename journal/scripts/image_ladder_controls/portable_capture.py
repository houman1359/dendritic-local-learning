#!/usr/bin/env python3
"""Verify and recompute all forty optional MNIST checkpoint captures."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from portable_run import sha,scientific_signature,validate_runtime,dump

def source_hash(path,original,record):
    expected=original
    if record:
        values=json.loads(record.read_text())
        if values['original_sha256']!=original:raise ValueError('Original analysis source identity mismatch')
        expected=values['released_sha256']
    if sha(path)!=expected:raise ValueError('Analysis source byte mismatch')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint-root','study-root','runtime-root','dataset-root','output-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--runtime-release-inventory',type=Path);p.add_argument('--adapter-release-record',type=Path);p.add_argument('--capture-release-record',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();study=a.study_root.resolve();runtime=a.runtime_root.resolve();out=a.output_root.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be new or empty')
    protocol=json.loads((study/'protocol.json').read_text());inv=json.loads((study/'portable_scientific_inventory.json').read_text())
    required=inv['runtime_required_sha256']
    assert all(protocol['runtime_sha256'].get(k)==v for k,v in required.items())
    validate_runtime(runtime,dict(protocol,runtime_sha256=required),a.runtime_release_inventory)
    for name,digest in protocol['mnist_cache_sha256'].items():
        if sha(a.dataset_root/name)!=digest:raise ValueError('MNIST bytes changed: '+name)
    here=Path(__file__).resolve().parent
    source_hash(here/'run.py',inv['adapter_sha256'],a.adapter_release_record)
    source_hash(here/'capture.py',json.loads((study/'analysis_freeze.json').read_text())['sha256']['capture.py'],a.capture_release_record)
    sys.path.insert(0,str(runtime/'src'))
    import run
    run.RUNTIME=runtime
    import capture
    import torch
    torch.set_num_threads(1)
    def probe():
        root=a.dataset_root/'mnist/MNIST/raw'
        x=np.fromfile(root/'t10k-images-idx3-ubyte',dtype=np.uint8)[16:].reshape(-1,784)[:2048]
        y=np.fromfile(root/'t10k-labels-idx1-ubyte',dtype=np.uint8)[8:][:2048]
        return torch.tensor(x,dtype=torch.float32)/255,torch.tensor(y,dtype=torch.long)
    capture.probe=probe
    manifest=pd.read_csv(a.checkpoint_root/'checkpoint_manifest.tsv',sep='\t')
    assert len(manifest)==40 and not manifest.duplicated(['architecture','seed','checkpoint']).any()
    outputs=[]
    for rec in manifest.to_dict('records'):
        checkpoint=(a.checkpoint_root/rec['path']).resolve();config=(a.checkpoint_root/rec['config_path']).resolve()
        assert checkpoint.is_relative_to(a.checkpoint_root.resolve()) and config.is_relative_to(a.checkpoint_root.resolve())
        assert sha(checkpoint)==rec['sha256'] and sha(config)==rec['config_sha256']
        assert scientific_signature(json.loads(config.read_text()))==rec['scientific_config_sha256']
        rows,difference=capture.measure(config,checkpoint,device=a.device)
        outputs.extend(dict(architecture=rec['architecture'],seed=rec['seed'],checkpoint=rec['checkpoint'],checkpoint_sha256=rec['sha256'],forward_max_difference=difference,**row) for row in rows)
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(outputs).to_csv(out/'delivery_coordinate_capture.csv',index=False)
    dump(out/'capture_replay_audit.json',dict(status='complete',checkpoints=40,probe_examples=2048,device=a.device,
        runtime_commit=protocol['runtime_commit'],all_checkpoint_and_config_hashes_verified=True,source_analysis_sha256=sha(here/'capture.py')))
    print(out/'delivery_coordinate_capture.csv',flush=True)
if __name__=='__main__':main()
