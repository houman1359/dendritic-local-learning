"""Projector and experimental-configuration invariants for the added MNIST rung."""
import importlib.util
from pathlib import Path
import json
import torch
import yaml
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('image_controls',HERE/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)

def test_subtree_projection_matches_explicit_dictionary_and_keeps_soma():
    g=torch.Generator().manual_seed(8)
    d=torch.randn(7,18,generator=g,dtype=torch.float64)
    p=torch.randn(7,6,generator=g,dtype=torch.float64)
    s=torch.randn(7,2,generator=g,dtype=torch.float64)
    fields=torch.cat([p.reshape(7,2,3),d.reshape(7,2,9)],-1)
    dictionary=torch.zeros(12,3,dtype=torch.float64)
    for i in range(3): dictionary[i,i]=1;dictionary[3+3*i:6+3*i,i]=1
    expected=fields@dictionary@torch.linalg.pinv(dictionary)
    out=run.project_subtrees([d,p,s])
    observed=torch.cat([out[1].reshape(7,2,3),out[0].reshape(7,2,9)],-1)
    torch.testing.assert_close(expected,observed,atol=1e-14,rtol=1e-14)
    assert out[2] is s
    torch.testing.assert_close(run.project_subtrees(out)[0],out[0])
    assert torch.linalg.matrix_rank(dictionary)==3
    torch.testing.assert_close(dictionary.sum(1),torch.ones(12,dtype=torch.float64))

def test_k3_captures_at_least_the_uniform_component():
    fields=torch.randn(11,5,12,generator=torch.Generator().manual_seed(9),dtype=torch.float64)
    projected=run.project_subtrees([fields[...,3:].reshape(11,45),fields[...,:3].reshape(11,15),torch.zeros(11,5)])
    out=torch.cat([projected[1].reshape(11,5,3),projected[0].reshape(11,5,9)],-1)
    assert ((out.square().sum(-1)-12*fields.mean(-1).square())>=-1e-12).all()
    torch.testing.assert_close((fields-out).sum(-1),torch.zeros(11,5,dtype=torch.float64),atol=1e-14,rtol=0)

def test_frozen_development_is_paired_full_dataset_and_equal_rate_budget():
    records=json.loads((run.OUT/'development_conditions.json').read_text())
    assert len(records)==90
    signatures={}
    for record in records:
        cfg=yaml.safe_load(Path(record['config']).read_text())
        assert cfg['training']['main']['common']['epochs']==180
        assert cfg['training']['main']['common']['batch_size']==256
        assert cfg['data']['dataset_name']=='mnist'
        assert cfg['training']['main']['common']['early_stopping'] is False
        cfg['outputs']={};cfg['_sweep_config_id']=None
        cfg['training']['main']['learning_strategy_config']['error_broadcast_mode']=None
        for key in ('lr','topk_lr','blocklinear_lr','reactivation_lr','decoder_lr'):
            cfg['training']['main']['common']['param_groups'][key]=round(cfg['training']['main']['common']['param_groups'][key]/record['multiplier'],10)
        signatures.setdefault((record['architecture'],record['seed']),set()).add(json.dumps(cfg,sort_keys=True))
    assert all(len(s)==1 for s in signatures.values())
