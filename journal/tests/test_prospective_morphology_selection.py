"""Numerical and protocol invariants for the prospective candidate selector."""
import importlib.util
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/run_prospective_morphology_selection.py'
spec=importlib.util.spec_from_file_location('prospective_tree_selection',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_protocol_has_disjoint_tasks_and_rotations():
    cfg=module.load()
    assert set(cfg['seeds']['development']).isdisjoint(cfg['seeds']['confirmatory'])
    assert set(cfg['angles_pi']['development']).isdisjoint(cfg['angles_pi']['confirmatory'])
    task=dict(seed=999001,rank=4,noise_sd=.75,angle_pi=.125)
    x,_,_=module.dataset(task,'calibration',256)
    tx,_,_=module.dataset(task,'training',256)
    assert not np.allclose(x,tx)


def test_laplacian_projection_moments_and_gradient(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'OUT',tmp_path)
    module.invariants()
    assert (tmp_path/'invariant_tests.json').exists()


def test_zero_coupling_full_budget_training_equivalence():
    cfg=module.load();cfg.update(joint_kappa=0.,n_training=128,n_test=128,training_steps=8,checkpoints=[8])
    candidates=[c for c in module.candidates(cfg) if c['budget_k']==8]
    assert all(np.allclose(c['transfer'],np.eye(8)) for c in candidates)
    task=dict(task_id='numerical_test',seed=999002,split='development',rank=4,noise_sd=.75,angle_pi=.125)
    a=module.train_task(task,'feedback_only',candidates,cfg)
    b=module.train_task(task,'joint_forward_feedback',candidates,cfg)
    assert np.allclose([r['test_loss'] for r in a],[r['test_loss'] for r in b],rtol=1e-13,atol=1e-13)
    assert np.ptp([r['test_loss'] for r in a])<1e-13
