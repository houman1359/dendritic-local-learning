#!/usr/bin/env python3
"""Verify archive provenance and derive portable MNIST replay hash arguments."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from portable_run import sha,dump,condition_signature,scientific_signature
import yaml
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--release-root',type=Path,required=True);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    release=a.release_root.resolve();study=a.study_root.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(HERE.parents[1]/'code/release_noise'))
    from release_hashes import verify_released_file
    protocol=json.loads((study/'protocol.json').read_text());freeze=json.loads((study/'freeze.json').read_text());inv=json.loads((study/'portable_scientific_inventory.json').read_text())
    def verify(path,original):
        verdict=verify_released_file(path,original)
        if not verdict['verified']:raise ValueError(f'Unverified release input {path}: {verdict["reason"]}')
        return verdict
    verify(study/'protocol.json',freeze['protocol_sha256'])
    with (release/'RELEASED_SOURCE_HASHES.tsv').open() as handle:records={row['path']:row for row in csv.DictReader(handle,delimiter='\t')}
    runtime=release/'historical_runtimes/image_ladder_6c1aaa2';rows=[]
    for name,original in inv['runtime_required_sha256'].items():
        if protocol['runtime_sha256'].get(name)!=original:raise ValueError('Runtime allowlist changed sourceidentity')
        file=runtime/name;key=str(file.relative_to(release));record=records[key]
        if record['origin_commit']!=protocol['runtime_commit'] or record['origin_sha256']!=original:raise ValueError('Runtime manifest origin mismatch')
        verdict=verify(file,original)
        rows.append(dict(path=name,original_sha256=original,released_sha256=verdict['released_sha256']))
    with (out/'runtime.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','original_sha256','released_sha256'],delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    source_expected={'adapter':('run.py',inv['adapter_sha256']),'capture':('capture.py',json.loads((study/'analysis_freeze.json').read_text())['sha256']['capture.py'])}
    if inv.get('projected_k1_adapter_sha256'):
        source_expected['projected_k1']=('projected_k1.py',inv['projected_k1_adapter_sha256'])
        addendum=json.loads((study/'projected_k1/freeze.json').read_text())
        verify(study/'projected_k1/protocol.json',addendum['protocol_sha256'])
    for kind,(name,original) in source_expected.items():
        verdict=verify(HERE/name,original)
        dump(out/f'{kind}.json',dict(original_sha256=original,released_sha256=verdict['released_sha256']))
    configs=0
    for stage in sorted(p.name.removesuffix('_conditions.json') for p in study.glob('*_conditions.json')):
        path=study/f'{stage}_conditions.json'
        if not path.exists():continue
        for rec in json.loads(path.read_text()):
            file=study/'configs'/stage/f'condition_{rec["index"]:03d}.yaml';verify(file,rec['config_sha256'])
            signature=scientific_signature(yaml.safe_load(file.read_text()));key=f'{stage}/condition_{rec["index"]:03d}'
            if signature!=inv['config_scientific_sha256'][key]:raise ValueError('Scientific configuration identity changed')
            if condition_signature(stage,rec,signature)!=inv['condition_scientific_sha256'][key]:raise ValueError('Scientific condition or delivery rule changed')
            configs+=1
    dump(out/'verification.json',dict(status='passed',runtime_files=len(rows),configs_verified=configs,runtime_commit=protocol['runtime_commit'],
        source_manifest_sha256=sha(release/'RELEASED_SOURCE_HASHES.tsv'),protocol_original_sha256=freeze['protocol_sha256']))
    print(json.dumps(dict(runtime_root=str(runtime),runtime_release_inventory=str(out/'runtime.tsv'),adapter_release_record=str(out/'adapter.json'),capture_release_record=str(out/'capture.json'),projected_k1_release_record=str(out/'projected_k1.json')),indent=2))
if __name__=='__main__':main()
