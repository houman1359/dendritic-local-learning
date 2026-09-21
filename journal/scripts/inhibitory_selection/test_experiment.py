from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parent))
from experiment import SelectionNet,dataset


def test_separable_target_and_unchanged_ood_labels():
    x,i,y=dataset(42,'ood',31,'separable',1.)
    xx,ii,yy=dataset(42,'ood',31,'separable',3.)
    torch.testing.assert_close(y,yy,rtol=0,atol=0)
    z=x[:,:8].log().reshape(-1,4,2)
    expected=(x[:,-4:] * .5*z.tanh().sum(-1)*torch.tensor([1,-1,1,-1])).sum(-1)
    torch.testing.assert_close(y,expected)
    assert not torch.equal(x,xx)


def test_nonlinear_parent_supports_feature_interactions():
    z=torch.tensor([[.3,-.7,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
    c=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
    def mixed(net):
        def f(a,b):
            v=z.clone();v[0,0]+=a;v[0,1]+=b
            s=torch.cat([v.exp(),(-v).exp()],-1)
            return net(torch.cat([s,c],-1),torch.cat([s,4*(1-c)],-1)).item()
        return f(.4,.4)-f(.4,0)-f(0,.4)+f(0,0)
    with torch.no_grad():
        assert abs(mixed(SelectionNet(2,variant='separable').double()))<1e-12
        assert abs(mixed(SelectionNet(2,variant='interaction').double()))>1e-9


@pytest.mark.parametrize('variant',['separable','interaction'])
def test_exact_gradient_and_local_forward(variant):
    net=SelectionNet(3,variant=variant).double()
    x,i,y=dataset(13,'train',17,variant)
    expected=net(x,i).detach()
    net.gradients(x,i,y,y.var(), 'exact')
    p=net.core.branch_layers[1].branches_to_output.log_weight
    actual=p.grad[0,0].item()
    with torch.no_grad():
        original=p[0,0].item();values=[]
        for delta in [1e-6,-1e-6]:
            p[0,0]=original+delta
            values.append(float(.5*(net(x,i)-y).square().mean()/y.var()))
        p[0,0]=original
    assert actual==pytest.approx((values[0]-values[1])/2e-6,rel=1e-5,abs=1e-9)
    net.local=True
    torch.testing.assert_close(net(x,i),expected,rtol=0,atol=0)
