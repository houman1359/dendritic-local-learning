"""Trainable compact Boolean shunting control with explicit positivity.

This ordinary circuit is a control for the production E/I parameterization.
It has not replaced an arm in the frozen production-transfer study. Equality
of the initial Boolean function does not imply equal optimization trajectories.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .interaction_shunting_fold import compact_shunting_flat


class PositiveDenominatorCircuit(nn.Module):
    """Rectified affine ratios and a positive-weight ReLU soma.

    v=z/(1+||z||_1) parameterizes every strict interior point ||v||_1<1.
    Thus denominators stay positive on [-1,1]^s in exact arithmetic. The
    inverse is z=v/(1-||v||_1); there is no learned denominator intercept.
    """
    def __init__(self,raw,width,*,dtype=torch.float32,device=None):
        super().__init__()
        raw=torch.as_tensor(raw,dtype=torch.long,device=device)
        if raw.ndim!=2 or raw.shape[0]!=14*width or raw.shape[1]<1:
            raise ValueError('Expected fourteen local branches per soma')
        self.register_buffer('indices',raw)
        sites,s=raw.shape
        self.numerator_weight=nn.Parameter(torch.zeros(sites,s,dtype=dtype,device=device))
        self.numerator_bias=nn.Parameter(torch.zeros(sites,dtype=dtype,device=device))
        self.denominator_preweight=nn.Parameter(torch.zeros(sites,s,dtype=dtype,device=device))
        self.coupling_preweight=nn.Parameter(torch.zeros(width,14,dtype=dtype,device=device))
        self.soma_bias=nn.Parameter(torch.zeros(width,dtype=dtype,device=device))
        self.readout=nn.Linear(width,1,dtype=dtype,device=device)

    def denominator_weight(self):
        z=self.denominator_preweight
        return z/(1+z.abs().sum(-1,keepdim=True))

    def forward(self,x):
        local=x[:,self.indices]
        numerator=(local*self.numerator_weight).sum(-1)+self.numerator_bias
        denominator=1+(local*self.denominator_weight()).sum(-1)
        features=numerator.relu()/denominator
        soma=(features.reshape(len(x),-1,14)*F.softplus(self.coupling_preweight)).sum(-1)+self.soma_bias
        return self.readout(soma.relu())


def from_production(production,raw_supports,d):
    """Initialize a trainable compact circuit to the production Boolean function."""
    folded,inventory=compact_shunting_flat(production,raw_supports,d)
    weight=folded.numerator_weight;width=production.readout.in_features
    model=PositiveDenominatorCircuit(raw_supports,width,dtype=weight.dtype,device=weight.device)
    with torch.no_grad():
        v=folded.denominator_weight;margin=1-v.abs().sum(-1,keepdim=True)
        if not torch.all(margin>0):raise ValueError('Nonpositive denominator margin')
        model.denominator_preweight.copy_(v/margin)
        model.numerator_weight.copy_(folded.numerator_weight)
        model.numerator_bias.copy_(folded.numerator_bias)
        c=folded.coupling
        model.coupling_preweight.copy_(c+torch.log(-torch.expm1(-c)))
        model.soma_bias.copy_(folded.soma_bias)
        model.readout.load_state_dict(folded.readout.state_dict())
    count=sum(p.numel() for p in model.parameters())
    assert count==inventory['actual_parameters']
    return model,{**inventory,'optimized_parameters':count,
        'denominator_parameterization':'v=z/(1+L1(z)); numerator unconstrained; soma coupling softplus',
        'scope':'Trainable ordinary rational control; matched initial Boolean function, unequal counts and optimizer geometry'}
