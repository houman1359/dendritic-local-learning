#!/usr/bin/env python3
"""Record portable scientific identities directly from original frozen configs."""
import json,sys
from pathlib import Path
from run import OUT,HERE,COMMIT,sha,dump
from portable_run import scientific_signature,condition_signature
import yaml

def main():
    mapping={};conditions={}
    for stage in sorted(p.name.removesuffix('_conditions.json') for p in OUT.glob('*_conditions.json')):
        manifest=OUT/f'{stage}_conditions.json'
        if not manifest.exists():continue
        for rec in json.loads(manifest.read_text()):
            p=Path(rec['config']);assert sha(p)==rec['config_sha256']
            key=f'{stage}/condition_{rec["index"]:03d}'
            mapping[key]=scientific_signature(yaml.safe_load(p.read_text()))
            conditions[key]=condition_signature(stage,rec,mapping[key])
    sys.path.insert(0,str(HERE.parent))
    from build_software_release import repository_file_allowed
    protocol=json.loads((OUT/'protocol.json').read_text())
    required={name:digest for name,digest in protocol['runtime_sha256'].items() if repository_file_allowed(Path(name),'implementation')}
    addendum_sha=sha(HERE/'projected_k1.py') if (OUT/'projected_k1/protocol.json').exists() else None
    dump(OUT/'portable_scientific_inventory.json',dict(runtime_commit=COMMIT,adapter_sha256=sha(HERE/'run.py'),projected_k1_adapter_sha256=addendum_sha,runtime_required_sha256=required,
        signature_rule='SHA256 of canonicalJSON sortedkeys, compactseparators; remove only data.base_dir, outputs, _sweep_config_id',
        config_scientific_sha256=mapping,condition_scientific_sha256=conditions,
        condition_signature_rule="SHA256 of stage/index/architecture/arm/multiplier/seed and configuration signature; I/O locations excluded"))
    protocol=json.loads((OUT/'protocol.json').read_text())
    dump(OUT/'runtime_origin.json',dict(commit=COMMIT,description='Clean historical source used by all new MNIST dictionary/LR/decodercontrols; source matches archived flagshiptrainer',
        files=required,runtime_file_count=len(required),allowlist='build_software_release.repository_file_allowed(scope=implementation); excludes unrelated projectrunners/configs/tests',full_source_inventory='protocol.json preserves complete historicalsourceidentity but unusedrunners are not mandatoryreleaseinputs',adapter='scripts/image_ladder_controls/run.py'))
    print('Prepared',len(mapping),'configuration scientific identities and',len(required),'allowlistedruntimefiles')
if __name__=='__main__':main()
