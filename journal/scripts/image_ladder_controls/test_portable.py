"""Release relocation must preserve the scientific model and byte identities."""
import copy,hashlib,importlib.util,json
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mnist_portable',HERE/'portable_run.py')
portable=importlib.util.module_from_spec(spec);spec.loader.exec_module(portable)

def test_relocation_preserves_science_but_rate_or_seed_changes_do_not():
    config={'data':{'base_dir':'old-data','dataset_name':'mnist'},'outputs':{'results_dir':'old-output','run_name':'old'},'_sweep_config_id':'old-id','experiment':{'seed':10},'training':{'epochs':180,'lr':.001}}
    original=portable.scientific_signature(config)
    relocated=copy.deepcopy(config);relocated['data']['base_dir']='new-data';relocated['outputs']={'results_dir':'new-output','run_name':'new'};relocated['_sweep_config_id']='new-id'
    assert portable.scientific_signature(relocated)==original
    relocated['training']['lr']*=3
    assert portable.scientific_signature(relocated)!=original
    changed=copy.deepcopy(config);changed['experiment']['seed']+=1
    assert portable.scientific_signature(changed)!=original

def test_runtime_transformation_chain_requires_original_and_actual_released_bytes(tmp_path):
    file=tmp_path/'src/test.py';file.parent.mkdir();file.write_text('original = 1\n')
    original=portable.sha(file)
    protocol={'runtime_sha256':{'src/test.py':original}}
    assert portable.validate_runtime(tmp_path,protocol)==1
    file.write_text('original = 1  # relocated documentation\n')
    with pytest.raises(ValueError,match='byte mismatch'):portable.validate_runtime(tmp_path,protocol)
    inventory=tmp_path/'release.tsv';inventory.write_text('path\toriginal_sha256\treleased_sha256\nsrc/test.py\t'+original+'\t'+portable.sha(file)+'\n')
    assert portable.validate_runtime(tmp_path,protocol,inventory)==1
    inventory.write_text(inventory.read_text().replace(original,'f'*64))
    with pytest.raises(ValueError,match='identity mismatch'):portable.validate_runtime(tmp_path,protocol,inventory)


def test_dispatch_rule_is_bound_even_when_trainer_config_is_identical():
    record=dict(index=2,architecture='shunting',arm='subtree_k3',multiplier=1.,seed=50300,config='/old/config.yaml',results_dir='/old/results')
    original=portable.condition_signature('fresh',record,'same trainer configuration')
    moved=dict(record,config='/new/config.yaml',results_dir='/new/results')
    assert portable.condition_signature('fresh',moved,'same trainer configuration')==original
    for field,value in [('arm','exact_path'),('seed',50301),('architecture','additive'),('multiplier',3.),('index',3)]:
        changed=dict(record);changed[field]=value
        assert portable.condition_signature('fresh',changed,'same trainer configuration')!=original
