"""Target-independent initialization of signed DendriNet from its Boolean fold.

The production object and its registered parameter inventory are preserved.
This mapping concerns a TWO-stage sparse ReLU circuit on Rademacher inputs.
It is not a mapping of the winning one-hidden-layer local ReLU bank.
"""
import torch

from .interaction_relu_controls import GroupedReLU
from .local_composition import production_population
from .order_spectrum import sign_split_contacts


def initialize_signed_from_compact(production, compact):
    """Lift a grouped compact circuit into existing signed production parameters.

    Each raw weight becomes the pair (w,-w); gate gains are initialized to one
    and thresholds to minus the compact biases. Gains remain trainable. Equal
    initial functions do not imply identical optimization trajectories.
    """
    if not isinstance(compact.hidden, GroupedReLU):
        raise ValueError('The compact control must have a local stage and a nonlinear soma')
    layers=list(production_population(production).branch_layers)
    if len(layers)!=2 or any(layer.use_shunting for layer in layers):
        raise ValueError('Requires flat signed production with one branch stage and one soma')
    branch,soma=layers
    if (branch.branch_inhibition is not None or soma.branch_excitation is not None
            or soma.branch_inhibition is not None):
        raise ValueError('Unexpected contact banks')
    if any(layer.reactivation.__class__.__name__!='ParametricReLU' for layer in layers):
        raise ValueError('Requires the existing learned ReLU gates')
    local=compact.hidden.local
    width=production.readout.in_features;s=local.indices.shape[1]
    if local.weight.shape!=(14*width,s) or compact.readout.in_features!=width:
        raise ValueError('Production and compact widths must match')
    encoded=branch.branch_excitation.connection_indices
    # Both halves encode the same raw contacts in positive/negative channels.
    raw=local.indices.detach().cpu().numpy()
    d=int((encoded.reshape(-1,2*s)[:,s:]-encoded.reshape(-1,2*s)[:,:s])[0,0])
    expected=torch.as_tensor(sign_split_contacts(raw,d),device=encoded.device,dtype=encoded.dtype)
    if not torch.equal(encoded.reshape_as(expected),expected):
        raise ValueError('Production and compact raw supports must match before initialization')
    before=sum(p.numel() for p in production.parameters())
    with torch.no_grad():
        branch.branch_excitation.pre_w.copy_(torch.cat([local.weight,-local.weight],dim=1).reshape_as(branch.branch_excitation.pre_w))
        branch.reactivation.log_m.zero_();branch.reactivation.b.copy_(-local.bias)
        soma.branches_to_output.log_weight.copy_(compact.hidden.coupling)
        soma.reactivation.log_m.zero_();soma.reactivation.b.copy_(-compact.hidden.bias)
        production.readout.load_state_dict(compact.readout.state_dict())
    assert sum(p.numel() for p in production.parameters())==before
    return dict(production_parameters=before,compact_parameters=sum(p.numel() for p in compact.parameters()),
                raw_fan_in=s,width=width,gains_remain_trainable=all(layer.reactivation.log_m.requires_grad for layer in layers),
                scope='Matched initial Boolean function and contacts; parameterization and optimizer geometry differ')


def signed_branch_gradient_metric(weight,bias,gain,weight_gradient,bias_gradient):
    """Apply J J^T for the branch fold, without materializing a square matrix.

    The original Euclidean coordinates are (positive weights, negative
    weights, threshold, log gain). This is a gradient-flow identity, not an
    Adam update or an exact finite learning-rate trajectory equivalence.
    """
    s=weight.shape[-1]
    radial=(weight*weight_gradient).sum(-1)+bias*bias_gradient
    return (.5*gain.square()[...,None]*weight_gradient+weight*radial[...,None],
            (s/2+1)*gain.square()*bias_gradient+bias*radial)


def signed_soma_gradient_metric(coupling,bias,gain,coupling_gradient,bias_gradient):
    radial=(coupling*coupling_gradient).sum(-1)+bias*bias_gradient
    return (gain.square()[...,None]*coupling_gradient+coupling*radial[...,None],
            gain.square()*bias_gradient+bias*radial)
