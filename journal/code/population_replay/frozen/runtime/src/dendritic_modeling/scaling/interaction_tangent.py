"""Full initial tangent model of the one-hidden-layer sparse ReLU control.

This is f(theta0,x) + J(theta0,x) delta, including hidden weights, hidden
biases, head weights and head bias. It is an attribution control, not another
parameter-matched architecture: the fixed initial state is counted separately.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .interaction_fanin import construct as construct_nonlinear
from .interaction_relu_controls import LocalReLU


class InitialTangentReLU(nn.Module):
    def __init__(self, model):
        super().__init__()
        if not isinstance(model.hidden, LocalReLU) or model.readout.out_features != 1:
            raise ValueError("Requires the one-hidden-layer scalar-output local ReLU control")
        self.register_buffer("indices", model.hidden.indices.detach().clone())
        for name,value in [("weight",model.hidden.weight),("bias",model.hidden.bias),
                           ("head_weight",model.readout.weight),("head_bias",model.readout.bias)]:
            self.register_buffer(name+"0",value.detach().clone())
            self.register_parameter("delta_"+name,nn.Parameter(torch.zeros_like(value)))

    def forward(self, x):
        contacts=x[:,self.indices]
        z0=(contacts*self.weight0).sum(-1)+self.bias0
        h0=z0.relu()
        # PyTorch's ReLU derivative at exactly zero is zero.
        dz=(contacts*self.delta_weight).sum(-1)+self.delta_bias
        return (F.linear(h0,self.head_weight0,self.head_bias0)
                +F.linear(h0,self.delta_head_weight,self.delta_head_bias)
                +F.linear((z0>0).to(z0.dtype)*dz,self.head_weight0))


def construct(task, *, dtype=torch.float32):
    if task['family']!='local_relu' or task['mode']!='end_to_end':
        raise ValueError("Specify the full local ReLU origin model; tangent coefficients are all trained")
    origin,inv=construct_nonlinear(task,dtype=dtype)
    model=InitialTangentReLU(origin)
    coefficients=sum(p.numel() for p in model.parameters())
    fixed=sum(b.numel() for b in model.buffers() if b.is_floating_point())
    assert coefficients==fixed==inv['actual_parameters']
    inventory=dict(origin_inventory=inv,origin_network_parameters=coefficients,
        optimized_tangent_coefficients=coefficients,fixed_initialization_scalars=fixed,
        stored_floating_scalars=coefficients+fixed,integer_contact_entries=model.indices.numel(),
        integer_contact_bytes=model.indices.numel()*model.indices.element_size(),
        tangent_parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
        scope='Full fixed initial Jacobian control, not a matched-total-storage nonlinear architecture')
    return model,inventory
