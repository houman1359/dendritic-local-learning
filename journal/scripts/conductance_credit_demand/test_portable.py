"""Portable integrity and scope checks; no new scientific outcomes."""
from pathlib import Path
from types import SimpleNamespace
import csv
import importlib.util
import json
import shutil
import sys
import pytest

HERE=Path(__file__).resolve().parent;JOURNAL=HERE.parents[1]
spec=importlib.util.spec_from_file_location('conductance_portable_tested',HERE/'portable_run.py');portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)


def fixture_journal(tmp_path):
    root=tmp_path/'journal';study=root/portable.DATA_PREFIX
    paths=['code/release_noise/release_hashes.py','source_data/conductance_credit_demand/science_handoff_inventory_20260906.tsv']
    for family in ['', 'opponent/']:
        prefix='source_data/conductance_credit_demand/'+family
        for name in ['development_freeze.json','protocol.json','selection_freeze.json']:
            paths.append(prefix+name)
        frozen=json.loads((JOURNAL/prefix/'development_freeze.json').read_text())
        paths.extend(frozen['source_sha256'])
    paths += ['scripts/conductance_credit_demand/extend.py','scripts/conductance_credit_demand/extend_opponent.py','source_data/conductance_credit_demand/opponent/extension_source_freeze.json']
    for relative in set(paths):
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(JOURNAL/relative,dest)
    return root,study

def mark_transformation(root,relative,original):
    changed=root/relative;released=portable.sha(changed);manifest=root/'transformation.tsv'
    fields=['original_source','original_sha256','sha256','transformation']
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t');writer.writeheader();writer.writerow(dict(original_source=relative,original_sha256=original,sha256=released,transformation='portable text normalization'))
    helper=portable.load_module('_test_portable_release_helper',root/'code/release_noise/release_hashes.py')
    helper.write_sidecar(root,[dict(path=relative,kind='source_data',origin_sha256=original,release_sha256=released,transformation='portable text normalization',provenance_file=manifest.name,provenance_sha256=portable.sha(manifest))],'source_data')


def test_both_family_protocols_and_original_environment_are_verified(tmp_path):
    root,study=fixture_journal(tmp_path)
    for family in ['first','opponent']:
        context=portable.input_context(study,root,family)
        assert context['freeze']['numpy']=='2.2.6'
        assert context['cfg']['steps']==4096
        assert len(context['selection']['confirmatory_tasks'])==2
        assert all(row['original_sha256']==row['released_sha256'] for row in context['verified'])


def test_original_to_released_protocol_chain_and_tamper_rejection(tmp_path):
    root,study=fixture_journal(tmp_path);relative='source_data/conductance_credit_demand/opponent/protocol.json';path=root/relative;original=portable.sha(path)
    path.write_bytes(path.read_bytes()+b'\n');mark_transformation(root,relative,original)
    context=portable.input_context(study,root,'opponent')
    record=next(r for r in context['verified'] if r['path']==relative)
    assert record['original_sha256']!=record['released_sha256']
    assert 'verified declared release transformation'==record['reason']
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='Unverified frozen input'):portable.input_context(study,root,'opponent')


def test_scientific_source_tamper_is_rejected(tmp_path):
    root,study=fixture_journal(tmp_path);source=root/'scripts/conductance_credit_demand/opponent_model.py';source.write_bytes(source.read_bytes()+b'\n# undeclared change\n')
    with pytest.raises(ValueError,match='Unverified frozen input'):portable.input_context(study,root,'opponent')


def test_full_and_excluded_budgets_preserve_settings():
    context=portable.input_context(JOURNAL/portable.DATA_PREFIX,JOURNAL,'opponent')
    cfg,phase,steps,tasks,budget=portable.execution_plan(context,'fresh',2101)
    assert cfg==context['cfg'] and phase=='fresh' and steps==budget==4096 and len(tasks)==2
    cfg,phase,steps,tasks,budget=portable.execution_plan(context,'extension',2101)
    assert steps==budget==16384 and cfg['checkpoints']==[0,64,256,1024,2048,4096,8192,12288,16384]
    cfg,phase,steps,tasks,budget=portable.execution_plan(context,'fresh',2101,4)
    assert cfg==context['cfg'] and steps==4 and budget==4096
    with pytest.raises(ValueError):portable.execution_plan(context,'fresh',999)
    with pytest.raises(ValueError):portable.execution_plan(context,'fresh',2101,4096)


def test_existing_output_is_never_overwritten(tmp_path):
    out=tmp_path/'output';out.mkdir();sentinel=out/'keep.txt';sentinel.write_text('unchanged')
    args=SimpleNamespace(study_root=JOURNAL/portable.DATA_PREFIX,journal_root=JOURNAL,family='opponent',phase='fresh',seed=2101,excluded_smoke_steps=1,verify_only=False,output_root=out)
    with pytest.raises(FileExistsError):portable.execute(args)
    assert sentinel.read_text()=='unchanged'


def test_declared_relocation_cannot_change_scientific_configuration(tmp_path):
    import subprocess
    root,study=fixture_journal(tmp_path);relative='source_data/conductance_credit_demand/opponent/protocol.json';path=root/relative;original=portable.sha(path);value=json.loads(path.read_text());value['n_train']+=1;path.write_text(json.dumps(value));mark_transformation(root,relative,original)
    # The byte transformation chain alone is valid, but the verified runner's
    # scientific defaults must still agree with the released configuration.
    result=subprocess.run([sys.executable,str(HERE/'portable_run.py'),'--study-root',str(study),'--journal-root',str(root),'--family','opponent','--seed','2101','--verify-only'],text=True,capture_output=True)
    assert result.returncode!=0
    assert 'Released protocol differs from the verified original scientific defaults' in result.stderr
