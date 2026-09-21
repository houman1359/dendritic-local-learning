#!/usr/bin/env python3
"""Separate optional archive for the forty checkpoints behind capture panels."""
from pathlib import Path
import csv,copy,hashlib,io,json,zipfile
import pandas as pd
from run import OUT,RAW,sha,dump
from portable_run import scientific_signature

def main():
    selected=pd.DataFrame(json.loads((OUT/'selection.json').read_text())['selected'])
    conditions=pd.DataFrame(json.loads((OUT/'fresh_conditions.json').read_text()))
    records=conditions[conditions.arm.eq('exact_path')].merge(selected[['architecture','arm','multiplier']],on=['architecture','arm','multiplier']).sort_values(['architecture','seed'])
    assert len(records)==20
    target=RAW/'exports/MNIST_Capture_Checkpoints.zip';target.parent.mkdir(parents=True,exist_ok=True)
    members=[];rows=[]
    for rec in records.to_dict('records'):
        root=Path(rec['results_dir']);audit=json.loads((root/'run_audit.json').read_text());results=Path(audit['model_results_dir'])
        cfg=json.loads((results/'config.json').read_text());original=copy.deepcopy(cfg)
        cfg['data']['base_dir']='DATA_ROOT'
        cfg['outputs']={'run_name':'mnist_capture_checkpoint','results_dir':'REPLAY_OUTPUT'}
        cfg.pop('_sweep_config_id',None)
        assert scientific_signature(cfg)==scientific_signature(original)
        prefix=f"checkpoints/{rec['architecture']}/seed_{rec['seed']}"
        config_bytes=(json.dumps(cfg,sort_keys=True,indent=2)+'\n').encode()
        config_name=prefix+'/config.json'
        members.append((config_name,config_bytes))
        for state,path in [('initial',root/'initial_model.pt'),('trained',results/'final_model.pt')]:
            name=prefix+'/'+state+'_model.pt';members.append((name,path))
            rows.append(dict(architecture=rec['architecture'],seed=rec['seed'],checkpoint=state,multiplier=rec['multiplier'],path=name,sha256=sha(path),bytes=path.stat().st_size,
                config_path=config_name,config_sha256=hashlib.sha256(config_bytes).hexdigest(),original_config_sha256=sha(results/'config.json'),scientific_config_sha256=scientific_signature(cfg)))
    frame=pd.DataFrame(rows);assert len(frame)==40
    manifest=frame.to_csv(index=False,sep='\t').encode();members.append(('checkpoint_manifest.tsv',manifest))
    readme='''# Optional MNIST capture checkpoints\n\nThis archive contains the 20 initial and 20 validation-selected exact-path\ncheckpoints behind Supplementary Fig. S49F. They come from the ten fresh\npaired seeds per architecture at development-selected rates. No test-based\nselection was performed. The complete config accompanies each seed; only\ndataset/output paths were replaced by portable placeholders.\n\nThe manifest lists every checkpoint hash and the original/released config\nhashes, plus a scientific config hash that excludes only I/O fields. The\nweights are unchanged binary copies. Keep this optional archive separate\nfrom the compact Source Data and source-code archives.\n\nAfter extracting this archive and the reviewer software, run:\n\n    python journal_package/journal/scripts/image_ladder_controls/portable_capture.py \\\n      --checkpoint-root /path/to/extracted/archive \\\n      --study-root /path/to/source_data/image_ladder_controls \\\n      --runtime-root /path/to/historical_mnist_runtime \\\n      --dataset-root /path/to/data \\\n      --output-root /path/to/new_capture_output --device cuda\n\nCPU execution is also supported. The script verifies checkpoint, config,\nruntime and official MNIST hashes and uses all 2,048 fixed test examples.\nSee the software README for declared original-to-released source hash\nrecords if the distribution sanitized source-code I/O paths.\n'''
    members.append(('README.md',readme.encode()))
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,value in members:
            if isinstance(value,Path):archive.write(value,name)
            else:archive.writestr(name,value)
    with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
    frame.to_csv(OUT/'capture_checkpoint_manifest.tsv',sep='\t',index=False)
    dump(OUT/'capture_checkpoint_archive.json',dict(filename=target.name,sha256=sha(target),bytes=target.stat().st_size,checkpoint_count=40,config_count=20,
        purpose='Optional independent replay of the complete exact-checkpoint capture analysis; separate from compactsoftwarearchive',
        weights='Byteidentical copies of initial and validation-selected modelstates',config_transformation='Onlydata.base_dir,outputs,and_sweep_config_idrelocated; scientificconfigurationhashunchanged'))
    print(target,flush=True)
if __name__=='__main__':main()
