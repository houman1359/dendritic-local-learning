"""Matched credit-rule adapters; existing forward models remain unchanged."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# The archived algebraic experiment imports its structure module as `model`.
# Loading it first preserves that established API; conductance is namespaced.
algebra = load('bridge_algebra_reference', HERE.parent/'morphology_credit/experiment.py')
structure = sys.modules['model']
conductance = load('bridge_conductance_reference', HERE.parent/'morphology_conductance/model.py')

RULES = ('exact', 'unit_broadcast', 'calibrated_broadcast', 'sign_broadcast')
FAMILIES = ('matching', 'quartet', 'nested')

def algebra_task(seed, family):
    """Matching and quartet have identical trees, leaves and input spectra."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 801]))
    permutation = rng.permutation(8)
    signs = rng.choice([-1., 1.], size=4)
    supports = ([permutation[j:j+2] for j in range(0,8,2)] if family == 'matching'
                else [permutation[:4], permutation[4:]] if family == 'quartet'
                else [permutation[:j] for j in (2,4,6,8)])
    coeff = np.zeros(256)
    for support, sign in zip(supports, signs):
        coeff[sum(1 << int(j) for j in support)] = .5 * sign
    if family in ('matching', 'quartet'):
        tree = structure.build_tree('compatible_shared_balanced', 'balanced', tuple(permutation))
    else:
        tree, bound, _ = algebra.tree_from_coeff(coeff, 'compatible_nested')
        assert bound == 0.
    assert structure.cut_scores(coeff, tree)['centered_cut_bound'] < 1e-20
    return coeff, tree

def algebra_state(theta, x, left, right):
    values = algebra.forward(x, theta, left, right)
    count = len(theta); ix = np.arange(count)
    q = np.zeros_like(values); q[:,14] = 1.
    features = np.zeros((count,7,len(x),4))
    for k in range(6,-1,-1):
        l = values[ix,left[:,k]]; r = values[ix,right[:,k]]
        _,b,c,d = theta[:,k,:].T
        q[ix,left[:,k]] = q[:,k+8] * (b[:,None] + d[:,None]*r)
        q[ix,right[:,k]] = q[:,k+8] * (c[:,None] + d[:,None]*l)
        features[:,k] = np.stack([np.ones_like(l),l,r,l*r],axis=-1)
    return dict(output=values[:,14], path=q[:,8:15].transpose(0,2,1), features=features)

def deliver(paths, profiles, rules):
    """Root remains one. All frozen profiles use only initial unlabeled inputs."""
    result = paths.copy()
    for i, rule in enumerate(rules):
        if rule == 'unit_broadcast': result[i,:,:6] = 1.
        elif rule == 'calibrated_broadcast': result[i,:,:6] = profiles[i]
        elif rule == 'sign_broadcast': result[i,:,:6] = np.sign(profiles[i])
        else: assert rule == 'exact'
    return result

def algebra_grad(theta, x, y, left, right, variance, profiles, rules):
    state = algebra_state(theta,x,left,right)
    routed = deliver(state['path'],profiles,rules)
    residual = (state['output']-y[None])/variance
    gradient = np.einsum('cb,cbj,cjbf->cjf',residual,routed,state['features'])/len(x)
    return gradient, state

def conductance_grad(theta, x, y, groups, variance, profiles, rules):
    # The established rule-1 API applies any fixed six-site profile.
    mapped_profiles = profiles.copy()
    rule_ids = np.ones(len(theta),int)
    for i, rule in enumerate(rules):
        if rule == 'exact': rule_ids[i] = 0
        elif rule == 'unit_broadcast': mapped_profiles[i] = 1.
        elif rule == 'sign_broadcast': mapped_profiles[i] = np.sign(mapped_profiles[i])
        else: assert rule == 'calibrated_broadcast'
    return conductance.gradients(theta,x,y,groups,variance,mapped_profiles,rule_ids)

def field_metrics(paths, residual, routed, profiles):
    """All capture values are squared-energy ratios and exclude the root.

    Oracle projection diagnostics fit one amplitude per example. They are not
    training rules. `best_rank_one` additionally chooses its spatial direction
    from the current field and therefore is a full information upper bound.
    """
    result = []
    for q, error, actual, profile in zip(paths[:,:,:6],residual,routed[:,:,:6],profiles):
        record = {}
        for label, field, given in [('path',q,actual), ('credit',q*error[:,None],actual*error[:,None])]:
            energy = np.sum(field*field)
            denominator = max(float(energy),1e-30)
            for name, direction in [('uniform',np.ones(6)),('calibrated',profile),('sign',np.sign(profile))]:
                norm = float(direction@direction)
                projection_energy = np.sum((field@direction)**2)/max(norm,1e-30)
                record[f'{label}_{name}_oracle_capture'] = float(projection_energy/denominator)
            spectrum = np.linalg.eigvalsh(field.T@field)
            record[f'{label}_best_rank_one_capture'] = float(spectrum[-1]/denominator)
            record[f'{label}_effective_rank'] = float(energy**2/max(float(spectrum@spectrum),1e-30))
            record[f'{label}_full_rank_capture'] = float(energy/denominator)
            record[f'{label}_delivered_relative_squared_error'] = float(np.sum((field-given)**2)/denominator)
            record[f'{label}_delivered_cosine'] = float(np.sum(field*given)/max(np.linalg.norm(field)*np.linalg.norm(given),1e-30))
            record[f'{label}_energy'] = float(energy)
        result.append(record)
    return result
