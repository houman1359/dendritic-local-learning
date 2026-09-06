"""Minimal directed shunting tree matching the paper's production branch balance.

Seven compartments: four terminal E/I compartments, two proximal compartments
with local inhibitory inputs, and a soma-only output. Identity reactivation,
leak=1, E_E=1, E_I=0. All conductances and input activities are nonnegative.
This is directed steady-state transport, not a reciprocal cable equation.
"""
from __future__ import annotations

import numpy as np

GROUPINGS = np.array([[[0,1],[2,3]], [[0,2],[1,3]], [[0,3],[1,2]]], dtype=int)
GROUP_NAMES = ['01_23', '02_13', '03_12']
NOMINAL_G = np.array([2.]*4 + [.7]*4 + [3.]*2 + [3.]*4 + [4.]*2)


def task_permutation(task_group):
    permutation = GROUPINGS[task_group].ravel()
    return np.concatenate([permutation, permutation+4, [8,9]])


def forward(log_g, x, groups):
    """Batched models, all receiving the same positive example matrix."""
    g = np.exp(log_g)
    models, examples = len(g), len(x)
    v = np.zeros((models, examples, 7))
    denominator = np.ones_like(v)
    e = g[:,None,:4]*x[None,:,:4]
    inhibitory = g[:,None,4:8]*x[None,:,4:8]
    denominator[:,:,:4] = 1+e+inhibitory
    v[:,:,:4] = e/denominator[:,:,:4]
    for p in range(2):
        children = np.take_along_axis(v[:,:,:4], groups[:,None,p,:], axis=2)
        coupling = g[:,10+2*p:12+2*p]
        denominator[:,:,4+p] = 1+coupling.sum(axis=1)[:,None]+g[:,8+p,None]*x[None,:,8+p]
        v[:,:,4+p] = np.sum(children*coupling[:,None,:], axis=2)/denominator[:,:,4+p]
    denominator[:,:,6] = (1+g[:,14:16].sum(axis=1))[:,None]
    v[:,:,6] = np.sum(v[:,:,4:6]*g[:,None,14:16], axis=2)/denominator[:,:,6]
    paths = np.ones_like(v)
    paths[:,:,4:6] = g[:,None,14:16]/denominator[:,:,6,None]
    mi, ni = np.arange(models)[:,None], np.arange(examples)[None,:]
    for p in range(2):
        for slot in range(2):
            leaf = groups[:,p,slot]
            paths[mi,ni,leaf[:,None]] = paths[:,:,4+p]*g[:,10+2*p+slot,None]/denominator[:,:,4+p]
    return dict(voltage=v, denominator=denominator, path=paths, conductance=g, output=v[:,:,6])


def calibrate_profiles(log_g, x, groups):
    """Each student's own initial label-free mean path, frozen during training."""
    return forward(log_g,x,groups)['path'][:,:,:6].mean(axis=1)


def delivery(exact_paths, profiles, groups, rule_ids):
    """Rules:0 exact,1 calibrated fixed broadcast,2 proximal-subtree projection,
    3 one-profile projection. Projection coefficients use the current exact
    field and are explicitly oracle diagnostics, not learned local encoders.
    """
    result = exact_paths.copy()
    for rule in [1,2,3]:
        chosen = np.flatnonzero(rule_ids==rule)
        if not len(chosen):
            continue
        p = profiles[chosen]
        exact = exact_paths[chosen,:,:6]
        if rule==1:
            result[chosen,:,:6] = p[:,None,:]
        elif rule==3:
            coefficient = np.einsum('mk,mnk->mn',p,exact)/np.sum(p*p,axis=1)[:,None]
            result[chosen,:,:6] = coefficient[:,:,None]*p[:,None,:]
        else:
            approximation = np.zeros_like(exact)
            for parent in range(2):
                mask = np.zeros_like(p)
                mask[:,4+parent] = 1.
                mask[np.arange(len(chosen))[:,None],groups[chosen,parent,:]] = 1.
                address = p*mask
                coefficient = np.einsum('mk,mnk->mn',address,exact)/np.sum(address*address,axis=1)[:,None]
                approximation += coefficient[:,:,None]*address[:,None,:]
            result[chosen,:,:6] = approximation
    return result


def gradients(log_g, x, y, groups, loss_variance, profiles=None, rule_ids=None):
    state = forward(log_g,x,groups)
    v,d,g = state['voltage'],state['denominator'],state['conductance']
    path = state['path'] if rule_ids is None else delivery(state['path'],profiles,groups,rule_ids)
    delta = (state['output']-y[None,:])/loss_variance
    local = np.zeros((len(log_g),len(x),16))
    local[:,:,:4] = x[None,:,:4]*(1-v[:,:,:4])/d[:,:,:4]*g[:,None,:4]*path[:,:,:4]
    local[:,:,4:8] = -x[None,:,4:8]*v[:,:,:4]/d[:,:,:4]*g[:,None,4:8]*path[:,:,:4]
    local[:,:,8:10] = -x[None,:,8:10]*v[:,:,4:6]/d[:,:,4:6]*g[:,None,8:10]*path[:,:,4:6]
    for p in range(2):
        child = np.take_along_axis(v[:,:,:4],groups[:,None,p,:],axis=2)
        local[:,:,10+2*p:12+2*p] = (child-v[:,:,4+p,None])/d[:,:,4+p,None]*g[:,None,10+2*p:12+2*p]*path[:,:,4+p,None]
    local[:,:,14:16] = (v[:,:,4:6]-v[:,:,6,None])/d[:,:,6,None]*g[:,None,14:16]
    gradient = np.mean(delta[:,:,None]*local,axis=1)
    return gradient,state


def input_gradient(log_g,x,groups):
    state = forward(log_g,x,groups)
    v,d,g,r = state['voltage'],state['denominator'],state['conductance'],state['path']
    jacobian = np.zeros((len(log_g),len(x),10))
    jacobian[:,:,:4] = g[:,None,:4]*(1-v[:,:,:4])/d[:,:,:4]*r[:,:,:4]
    jacobian[:,:,4:8] = -g[:,None,4:8]*v[:,:,:4]/d[:,:,:4]*r[:,:,:4]
    jacobian[:,:,8:10] = -g[:,None,8:10]*v[:,:,4:6]/d[:,:,4:6]*r[:,:,4:6]
    return jacobian


def teacher_parameters(seed):
    rng = np.random.default_rng(seed+910_000)
    return np.log(NOMINAL_G)+rng.normal(0,.25,16)


def dataset(seed, task_group, kind, n):
    offsets={'calibration':11,'training':23,'validation':31,'test':47,'spectrum':59}
    rng=np.random.default_rng(np.random.SeedSequence([seed,offsets[kind]]))
    standard = rng.normal(size=(n,10))
    scales = np.array([.7]*4+[.7]*4+[1.2]*2)
    base = np.exp(standard*scales-.5*scales**2)
    permutation = task_permutation(task_group)
    x = base[:,np.argsort(permutation)]
    teacher = teacher_parameters(seed)[None]
    y = forward(teacher,base,GROUPINGS[[0]])['output'][0]
    # f_task(x)=f_canonical(x[:,permutation]); this coupled sampling preserves
    # the full input-gradient spectrum across task permutations exactly.
    return x,y


def teacher_gradient_in_task_coordinates(seed,task_group,x):
    permutation=task_permutation(task_group)
    canonical=input_gradient(teacher_parameters(seed)[None],x[:,permutation],GROUPINGS[[0]])[0]
    return canonical[:,np.argsort(permutation)]


def equivalent_teacher_parameters(seed,task_group):
    """Constructive teacher in the task-compatible student grouping."""
    canonical=teacher_parameters(seed)
    permutation=GROUPINGS[task_group].ravel()
    equivalent=canonical.copy()
    equivalent[permutation]=canonical[:4]
    equivalent[permutation+4]=canonical[4:8]
    return equivalent
