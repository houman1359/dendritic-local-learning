"""Mechanism tests: locality, eligibility and oracle bookkeeping, not accuracy tests."""
import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_gate_model_under_test',HERE/'model.py');local=importlib.util.module_from_spec(spec);spec.loader.exec_module(local)

def reference():
    root=HERE.parent/'conductance_credit_demand';old=sys.modules.get('model')
    spec=importlib.util.spec_from_file_location('model',root/'model.py');first=importlib.util.module_from_spec(spec);spec.loader.exec_module(first)
    sys.modules['model']=first
    try:
        spec=importlib.util.spec_from_file_location('released_opponent_for_gate_test',root/'opponent_model.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    finally:
        if old is None:sys.modules.pop('model',None)
        else:sys.modules['model']=old
    return ref

def fixture():
    x,y,_=local.data(2026090799,'diagnostic',64,1.,10.)
    rng=np.random.default_rng(884);theta=np.log(local.NOMINAL)[None]+rng.normal(0,.4,(1,24))
    p=local.exact_path(local.forward(theta,x))[:,:,:6].mean(1)
    return theta,x,y,p

def test_exact_adapter_agrees_with_released_equations():
    ref=reference();theta,x,y,p=fixture()
    np.testing.assert_array_equal(local.data(2026090799,'diagnostic',64,1.,10.)[0],ref.data(2026090799,'diagnostic',64,1.,10.)[0])
    np.testing.assert_array_equal(y,ref.data(2026090799,'diagnostic',64,1.,10.)[1])
    np.testing.assert_array_equal(local.forward(theta,x)['output'],ref.forward(theta,x)['output'])
    for rule in ['exact','unit_broadcast','calibrated_broadcast','ancestry_three_oracle']:
        a=local.gradients(theta,x,y,np.var(y),p,[rule])[0];b=ref.gradients(theta,x,y,np.var(y),p,[rule])[0]
        np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)

def test_exact_parameter_gradient_matches_finite_differences():
    theta,x,y,p=fixture();g=local.gradients(theta,x,y,np.var(y),p,['exact'])[0][0];fd=[];h=1e-6
    for j in range(24):
        d=np.zeros_like(theta);d[0,j]=h
        losses=[np.mean((local.forward(theta+s*d,x)['output'][0]-y)**2)/(2*np.var(y)) for s in [1,-1]]
        fd.append((losses[0]-losses[1])/(2*h))
    np.testing.assert_allclose(g,fd,rtol=1e-5,atol=2e-8)

@pytest.mark.parametrize('rule',sorted(local.LOCAL_RULES))
def test_local_rules_never_call_exact_paths(monkeypatch,rule):
    theta,x,y,p=fixture()
    def forbidden(*args):raise AssertionError('Local rule attempted oracle path evaluation')
    monkeypatch.setattr(local,'exact_path',forbidden)
    g,_=local.gradients(theta,x,y,np.var(y),np.full_like(p,np.nan),[rule])
    assert np.isfinite(g).all()

def test_distal_gate_preserves_proximal_eligibilities_and_all_parameter_slots():
    theta,x,y,p=fixture();state=local.forward(theta,x);e=local.eligibility(state,x);q=local.delivered(state,x,p,['hard_distal_unit_proximal'])
    np.testing.assert_array_equal(q[0,:,:4],np.repeat((x[:,8:10]==0).astype(float),2,axis=1))
    np.testing.assert_array_equal(q[0,:,4:],np.ones((len(x),3)))
    assert e.shape[-1]==24
    # Inhibitory-gain eligibility is nonzero exactly when its parent is inhibited.
    for parent in [0,1]:
        active=x[:,8+parent]>0
        assert np.all(e[0,active,16+parent]<0)
        assert np.all(e[0,~active,16+parent]==0)
    g,_=local.gradients(theta,x,y,np.var(y),p,['hard_distal_unit_proximal'])
    assert np.all(np.abs(g[0,16:18])>1e-12)

def test_gating_proximal_credit_freezes_inhibitory_gains_by_construction():
    theta,x,y,p=fixture();g,_=local.gradients(theta,x,y,np.var(y),p,['hard_distal_and_proximal'])
    np.testing.assert_array_equal(g[0,16:18],np.zeros(2))

def test_two_leaf_dictionary_keeps_unit_proximal_and_separates_oracle_access():
    theta,x,y,p=fixture();state=local.forward(theta,x)
    two=local.delivered(state,x,p,['ancestry_two_leaf_oracle_unit_proximal']);three=local.delivered(state,x,p,['ancestry_three_oracle'])
    np.testing.assert_array_equal(two[:,:,:4],three[:,:,:4]);np.testing.assert_array_equal(two[:,:,4:],np.ones_like(two[:,:,4:]))

def test_proportional_gate_uses_declared_inhibitory_open_conductance():
    theta,x,y,p=fixture();state=local.forward(theta,x);q=local.delivered(state,x,p,['shunt_proportional_unit_proximal'])
    expected=np.repeat(1/(1+np.exp(theta[0,16:18])[None]*x[:,8:10]),2,axis=1)
    np.testing.assert_array_equal(q[0,:,:4],expected)
