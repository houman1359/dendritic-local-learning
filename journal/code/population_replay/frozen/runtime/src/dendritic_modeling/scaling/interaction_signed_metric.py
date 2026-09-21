"""A diagnostic SGD control that removes the fixed-gain Boolean fold metric.

It uses gradients from the actual production module. With gate gains frozen,
the fold is affine, so its exact-arithmetic update equals ordinary compact
SGD, including at finite step size. This is not an Adam equivalence or a
performance claim, and it leaves the production's stored parameter count intact.
"""
import torch
from .local_composition import production_population


@torch.no_grad()
def compact_metric_sgd_step(model,learning_rate):
    if learning_rate<=0:raise ValueError('Positive learning rate required')
    layers=list(production_population(model).branch_layers)
    if len(layers)!=2 or any(layer.use_shunting for layer in layers):
        raise ValueError('Requires flat signed production')
    branch,soma=layers
    if (branch.branch_inhibition is not None or soma.branch_excitation is not None
            or soma.branch_inhibition is not None
            or branch.branch_excitation.weight_transform!='identity'
            or soma.branches_to_output.weight_transform!='identity'
            or any(layer.reactivation.__class__.__name__!='ParametricReLU' for layer in layers)):
        raise ValueError('Requires the qualified signed contacts and ReLU gates')
    if any(layer.reactivation.log_m.requires_grad for layer in layers):
        raise ValueError('Freeze both gate gains explicitly before this diagnostic')
    pre=branch.branch_excitation.pre_w
    s=pre.shape[-1]//2
    if pre.shape[-1]!=2*s or pre.grad is None or branch.reactivation.b.grad is None:
        raise ValueError('Missing complete sign-split branch gradients')
    indices=branch.branch_excitation.connection_indices
    offsets=indices[:,s:]-indices[:,:s]
    if not torch.all(offsets==offsets[0,0]) or offsets[0,0]<=0:
        raise ValueError('Contacts must pair raw coordinates before sign splitting')
    coupling=soma.branches_to_output.log_weight
    updated=[pre,branch.reactivation.b,coupling,soma.reactivation.b,*model.readout.parameters()]
    if ({id(p) for p in model.parameters() if p.requires_grad}!={id(p) for p in updated}
            or any(p.grad is None or not torch.isfinite(p.grad).all() for p in updated)):
        raise ValueError('Unexpected trainable parameter or missing/nonfinite gradient')
    g=branch.reactivation.log_m.exp()
    if not torch.all(torch.isfinite(g)&(g>0)):raise ValueError('Invalid branch gain')
    root_gain=soma.reactivation.log_m.exp()
    if not torch.all(torch.isfinite(root_gain)&(root_gain>0)):raise ValueError('Invalid soma gain')
    grad_weight=(pre.grad[:,:s]-pre.grad[:,s:])/g[:,None]
    grad_bias=-branch.reactivation.b.grad/g
    shared=grad_bias/(g*(s+2))
    direction=torch.cat([grad_weight/g[:,None]+shared[:,None],-grad_weight/g[:,None]+shared[:,None]],-1)
    pre.add_(direction,alpha=-learning_rate)
    branch.reactivation.b.add_(-2*shared,alpha=-learning_rate)
    coupling.add_(coupling.grad/root_gain[:,None].square(),alpha=-learning_rate)
    soma.reactivation.b.add_(soma.reactivation.b.grad/root_gain.square(),alpha=-learning_rate)
    for parameter in model.readout.parameters():parameter.add_(parameter.grad,alpha=-learning_rate)
