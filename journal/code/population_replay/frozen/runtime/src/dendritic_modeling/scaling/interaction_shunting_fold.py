"""Boolean-only compact rational representation of flat shunting DendriNet.

A checkpoint mapping, not a new trained arm in the frozen transfer campaign.
The denominator must remain positive; unconstrained training of this diagnostic
module is not a qualified optimization policy.
"""
import numpy as np
import torch
from torch import nn
from .local_composition import production_population
from .order_spectrum import sign_split_contacts


class CompactShuntingCircuit(nn.Module):
    def __init__(self,raw,width,*,dtype,device):
        super().__init__()
        self.register_buffer('indices',torch.as_tensor(raw,dtype=torch.long,device=device))
        sites,s=raw.shape
        self.numerator_weight=nn.Parameter(torch.zeros(sites,s,dtype=dtype,device=device))
        self.numerator_bias=nn.Parameter(torch.zeros(sites,dtype=dtype,device=device))
        self.denominator_weight=nn.Parameter(torch.zeros(sites,s,dtype=dtype,device=device))
        self.coupling=nn.Parameter(torch.ones(width,14,dtype=dtype,device=device))
        self.soma_bias=nn.Parameter(torch.zeros(width,dtype=dtype,device=device))
        self.readout=nn.Linear(width,1,dtype=dtype,device=device)

    def forward(self,x):
        local=x[:,self.indices]
        numerator=(local*self.numerator_weight).sum(-1)+self.numerator_bias
        denominator=1+(local*self.denominator_weight).sum(-1)
        branch=numerator.relu()/denominator
        soma=(branch.reshape(len(x),-1,14)*self.coupling).sum(-1)+self.soma_bias
        return self.readout(soma.relu())


def compact_shunting_flat(production,raw_supports,d):
    """Fold E/I banks, gates and the parameter-only soma denominator."""
    raw=np.asarray(raw_supports);width=production.readout.in_features;s=raw.shape[-1]
    layers=list(production_population(production).branch_layers)
    if len(layers)!=2 or not all(layer.use_shunting for layer in layers):
        raise ValueError('Requires actual flat shunting with one branch stage and one soma')
    branch,soma=layers
    if branch.branch_excitation is None or branch.branch_inhibition is None or soma.branch_excitation is not None or soma.branch_inhibition is not None:
        raise ValueError('Unexpected contact layout')
    if raw.shape!=(14*width,s):raise ValueError('Unexpected support shape')
    if any(layer.reactivation.__class__.__name__!='ParametricReLU' for layer in layers):
        raise ValueError('Requires positive-gain ReLU gates')
    parameter=next(production.parameters())
    expected=torch.as_tensor(sign_split_contacts(raw,d),dtype=torch.long,device=parameter.device)
    for bank in [branch.branch_excitation,branch.branch_inhibition]:
        if bank.weight_norm_order is not None:raise ValueError('Unqualified contact normalization')
        if not torch.equal(bank.connection_indices.reshape_as(expected),expected):raise ValueError('Mismatched raw contacts')
    model=CompactShuntingCircuit(raw,width,dtype=parameter.dtype,device=parameter.device)
    with torch.no_grad():
        e=branch.branch_excitation.sparse_weight().reshape(-1,2*s)
        i=branch.branch_inhibition.sparse_weight().reshape(-1,2*s)
        assert torch.all(e>0) and torch.all(i>0)
        e0=e.sum(-1)/2;ea=(e[:,:s]-e[:,s:])/2
        i0=i.sum(-1)/2;ia=(i[:,:s]-i[:,s:])/2
        d0=1+branch.epsilon+e0+i0;da=ea+ia
        gain=branch.reactivation.log_m.exp();threshold=branch.reactivation.b
        model.numerator_weight.copy_(gain[:,None]*(ea-threshold[:,None]*da)/d0[:,None])
        model.numerator_bias.copy_(gain*(e0-threshold*d0)/d0)
        model.denominator_weight.copy_(da/d0[:,None])
        c=soma.branches_to_output.weight();g=soma.reactivation.log_m.exp()
        model.coupling.copy_(g[:,None]*c/(1+soma.epsilon+c.sum(-1))[:,None])
        model.soma_bias.copy_(-g*soma.reactivation.b)
        model.readout.load_state_dict(production.readout.state_dict())
        margin=1-model.denominator_weight.abs().sum(-1)
        assert torch.all(margin>0) and torch.all(model.coupling>0)
    count=sum(p.numel() for p in model.parameters())
    assert count==(28*s+30)*width+1
    return model,dict(actual_parameters=count,production_parameters=sum(p.numel() for p in production.parameters()),
        width=width,raw_fan_in=s,minimum_cube_denominator_margin=float(margin.min()),
        scope='Exact Boolean checkpoint fold into a positive-soma rectified rational circuit; optimization not matched')
