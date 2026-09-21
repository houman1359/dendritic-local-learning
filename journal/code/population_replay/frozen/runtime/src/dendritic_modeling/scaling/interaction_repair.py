"""Controlled production-coordinate and operator interventions on Boolean inputs."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

from .interaction_networks import construct as production
from .interaction_relu_controls import construct as compact
from .interaction_production_transfer import initialize_signed_from_compact
from .interaction_initialization import initialize_gates
from .interaction_shunting_control import PositiveDenominatorCircuit, from_production
from .interaction_stream import BooleanStream
from .local_composition import production_population
from .order_spectrum import target_packet


ARMS = (
    'signed_learned_sgd', 'signed_fixed_sgd', 'signed_metric_sgd', 'compact_signed_sgd',
    'signed_learned_adam', 'signed_fixed_adam', 'compact_signed_adam',
    'shunting_adam', 'compact_shunting_adam', 'compact_shunting_matched_P_adam',
    'positive_soma_relu_adam', 'signed_soma_rational_adam',
)


class OperatorControl(PositiveDenominatorCircuit):
    """Change division or coupling signs explicitly, keeping the contact inventory."""
    def __init__(self, source, *, division, positive_soma):
        raw = source.indices
        super().__init__(raw, source.readout.in_features,
                         dtype=source.numerator_weight.dtype, device=raw.device)
        self.load_state_dict(source.state_dict())
        self.division, self.positive_soma = division, positive_soma
        if not division:
            self.denominator_preweight.requires_grad_(False)
        if not positive_soma:
            with torch.no_grad():
                self.coupling_preweight.copy_(F.softplus(source.coupling_preweight))

    def forward(self, x):
        local = x[:, self.indices]
        numerator = (local * self.numerator_weight).sum(-1) + self.numerator_bias
        features = numerator.relu()
        if self.division:
            features = features / (1 + (local * self.denominator_weight()).sum(-1))
        coupling = F.softplus(self.coupling_preweight) if self.positive_soma else self.coupling_preweight
        soma = (features.reshape(len(x), -1, 14) * coupling).sum(-1) + self.soma_bias
        return self.readout(soma.relu())


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def construct(task, *, device='cpu', dtype=torch.float32):
    arm = task['family']
    if arm not in ARMS:
        raise ValueError(arm)
    kwargs = dict(d=task['d'], s=task['s'], model_seed=task['model_seed'],
                  support_seed=task['support_seed'], dtype=dtype)
    ceiling = task['ceiling']
    signed = arm.startswith('signed_') and arm != 'signed_soma_rational_adam'
    signed = signed or arm.startswith('compact_signed_')
    if signed:
        native, ni = production('flat_signed', ceiling, **kwargs)
        folded, fi = compact('sparse_relu2', ceiling, width=ni['width'],
                             raw_supports=np.asarray(ni['raw_supports']), **kwargs)
        generator = torch.Generator().manual_seed(task['bias_seed'])
        with torch.no_grad():
            folded.hidden.local.bias.copy_(task['bias_scale'] * torch.randn(
                folded.hidden.local.bias.shape, generator=generator, dtype=dtype))
        mapping = initialize_signed_from_compact(native, folded)
        if arm.startswith('signed_fixed_') or arm == 'signed_metric_sgd':
            for layer in production_population(native).branch_layers:
                layer.reactivation.log_m.requires_grad_(False)
        model, inv = (folded, fi) if arm.startswith('compact_signed_') else (native, ni)
        model.to(device)
        inv.update(mapping=mapping, origin_parameters=ni['actual_parameters'],
                   compact_update_dimension=fi['actual_parameters'], unlabeled_initialization_draws=0)
    else:
        width = (ceiling - 1) // (28 * task['s'] + 30) if arm == 'compact_shunting_matched_P_adam' else None
        native, ni = production('flat_shunting', ceiling, width=width, **kwargs)
        native.to(device)
        dummy = target_packet(d=task['d'], rho=.5, seed=63101)
        unlabeled, _ = BooleanStream(dummy, task['unlabeled_seed'], device=device, dtype=dtype).next(4096)
        initialization = initialize_gates(native, unlabeled, 'center_scale_all')
        if arm == 'shunting_adam':
            model, inv = native, ni
        else:
            model, inv = from_production(native, np.asarray(ni['raw_supports']), task['d'])
            if arm in ['positive_soma_relu_adam', 'signed_soma_rational_adam']:
                model = OperatorControl(model, division=arm != 'positive_soma_relu_adam',
                                        positive_soma=arm != 'signed_soma_rational_adam')
        inv.update(origin_parameters=ni['actual_parameters'], gate_initialization=initialization,
                   unlabeled_initialization_draws=4096,
                   intervention_initial_function_preserved=arm != 'positive_soma_relu_adam',
                   intervention='remove_division' if arm == 'positive_soma_relu_adam' else
                                'allow_signed_soma' if arm == 'signed_soma_rational_adam' else 'coordinates_or_inventory')
    inv.update(study_arm=arm, budget_ceiling=ceiling,
               actual_parameters=sum(p.numel() for p in model.parameters()),
               optimized_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               parameter_shapes={n:list(p.shape) for n,p in model.named_parameters()},
               trainable_parameter_names=[n for n,p in model.named_parameters() if p.requires_grad],
               initial_state_hash=state_hash(model),
               scope='Development intervention; matched-function arms have unequal stored counts; no exponent claim')
    assert inv['actual_parameters'] <= ceiling
    return model, inv
