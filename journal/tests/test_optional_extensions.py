"""Mechanistic checks for prospective proxies, routing and temporal eligibility."""
import importlib.util
from pathlib import Path
import torch

J=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('extensions',J/'code/optional_extensions/experiments.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
torch.set_num_threads(1)


def test_inherited_forward_and_original_credit_are_unchanged():
    x,i,y=e.dataset(91,'train',64,'interaction');var=y.var(unbiased=False)
    for rule in ['resistance','derivative','exact']:
        old=e.RescueNet(19).double();new=e.ExtensionNet(19,'proxy',rule).double()
        torch.testing.assert_close(new(x,i),old(x,i),rtol=0,atol=0)
        old.gradients(x,i,y,var,rule);new.gradients(x,i,y,var)
        for p,q in zip(old.parameters(),new.parameters()):
            if p.grad is None:assert q.grad is None
            else:torch.testing.assert_close(q.grad,p.grad,rtol=0,atol=0)


def test_proxy_bins_and_context_controls_preserve_declared_marginals():
    v=torch.tensor([[-.9,-.2],[0.,.6],[.4,.8],[.7,-.3]],dtype=torch.float64)
    rng=torch.Generator().manual_seed(11)
    q=e.slope_proxy(v,'bins4',rng)
    expected=1-torch.tensor([[.875,.125],[.125,.625],[.375,.875],[.625,.375]],dtype=torch.float64).tanh().square()
    torch.testing.assert_close(q,expected,rtol=0,atol=0)
    # Both proxy resolutions must actually vary inside the physical range.
    grid=torch.linspace(0,.999,100,dtype=torch.float64)[:,None]
    assert e.slope_proxy(grid,'bins2',rng).unique().numel()==2
    assert e.slope_proxy(grid,'bins4',rng).unique().numel()==4
    c=torch.tensor([0,1,0,1]);shuffled=e.context_shuffle(q,c,rng);mean=e.context_mean(q,c)
    for context in [0,1]:
        torch.testing.assert_close(q[c==context].sort(0).values,shuffled[c==context].sort(0).values)
        torch.testing.assert_close(q[c==context].mean(0),mean[c==context].mean(0))


def test_proxy_randomization_does_not_change_data_or_global_rng():
    v=torch.zeros(1024,4,dtype=torch.float64)
    state=torch.get_rng_state();generator=torch.Generator().manual_seed(7)
    a=e.slope_proxy(v,'noise05',generator)
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    b=e.slope_proxy(v,'noise05',torch.Generator().manual_seed(7))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert a.var()>0 and bool(((a>=0)&(a<=1)).all())


def test_proxy_bins_span_reachable_parent_voltages():
    net=e.ExtensionNet(421,'proxy','bins4').double()
    for severity in [1.,3.]:
        x,i,y=e.dataset(116,'ood',512,'interaction',severity)
        with torch.no_grad():net(x,i)
        v=net.core.branch_layers[1]._last_branch_diagnostics['V']
        assert bool(((v>=0)&(v<1)).all())


def test_learned_router_has_fixed_dose_and_no_terminal_cue_bypass():
    x,i,y=e.dataset(17,'train',16,'interaction')
    net=e.ExtensionNet(23,'routing','learned_local_augmented').double()
    captured=[]
    hook=net.core.branch_layers[0].register_forward_pre_hook(lambda mod,args:captured.append(args[0].detach().clone()))
    net.gradients(x,i,y,y.var(unbiased=False));hook.remove()
    assert captured and captured[0][:,-4:].count_nonzero()==0
    torch.testing.assert_close(net.last_cue[:,-4:].sum(-1),torch.full((16,),12.,dtype=torch.float64))
    assert net.route_logits.grad is not None and net.route_logits.grad.norm()>0


def test_uniform_routing_has_no_hidden_forward_cue_channel():
    x,i,_=e.dataset(79,'test',1,'interaction')
    x=x.expand(4,-1).clone();i=i.expand(4,-1).clone();cue=torch.eye(4,dtype=torch.float64)
    x[:,-4:]=cue;i[:,-4:]=4*(1-cue)
    net=e.ExtensionNet(101,'routing','uniform_augmented').double()
    with torch.no_grad():prediction=net(x,i)
    torch.testing.assert_close(prediction,prediction[0].expand_as(prediction),rtol=0,atol=1e-15)


def test_online_memory_eligibility_equals_bptt_and_one_step_preserves_forward():
    rng=torch.Generator().manual_seed(109)
    x=torch.randn(11,8,8,dtype=torch.float64,generator=rng)
    a=torch.randn(8,dtype=torch.float64,generator=rng,requires_grad=True)
    g=(.1*torch.randn(8,dtype=torch.float64,generator=rng)).requires_grad_()
    outputs={m:e.memory_state(x,a,g,m) for m in ['trace','one_step','bptt']}
    for v in outputs.values():torch.testing.assert_close(v,outputs['bptt'],rtol=0,atol=0)
    gradients={m:torch.autograd.grad(v.square().sum(),(a,g),retain_graph=True) for m,v in outputs.items()}
    for actual,expected in zip(gradients['trace'],gradients['bptt']):torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
    assert (gradients['one_step'][0]-gradients['bptt'][0]).norm()>1e-3


def test_temporal_target_and_all_arms_have_finite_updates():
    for study,arms in e.ARMS.items():
        data=e.temporal_dataset(72,'train',32) if study=='temporal' else e.dataset(72,'train',32,'interaction')
        for arm in arms:
            net=e.ExtensionNet(32,study,arm).double()
            loss=net.gradients(*data,data[2].var(unbiased=False))
            assert loss>=0 and torch.isfinite(torch.tensor(loss))
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
