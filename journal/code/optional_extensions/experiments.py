"""Prospective optional extensions of the frozen population credit experiment.

The published execution sources are imported unchanged. New mechanisms live
here and are tested separately; no existing checkpoint or cohort is modified.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
J = HERE.parents[1]
spec = importlib.util.spec_from_file_location('population_launcher', J/'code/population_replay/launch.py')
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
FROZEN, IDENTITY, ORIGINAL = launcher.verified_sources()
sys.path[:0] = [str(FROZEN/p) for p in ['study','selection','base','runtime/src']]
from rescue import RescueNet, context_shuffle, active_values, bound_fraction
from experiment import dataset
from run import evaluate, sha
import dendritic_modeling
assert Path(dendritic_modeling.__file__).resolve().is_relative_to(FROZEN/'runtime')

PROXIES = ['resistance','derivative','exact','noise025','noise05','noise1',
           'bins2','bins4','shuffle_noise05','shuffle_bins4','mean_bins4']
ROUTES = ['oracle_augmented','learned_local_augmented','learned_exact_router_augmented',
          'learned_local_resistance','uniform_augmented','wrong_augmented']
TEMPORAL = ['exact_trace','augmented_trace','resistance_trace',
            'exact_one_step','augmented_one_step','exact_no_memory']
ARMS = dict(proxy=PROXIES, routing=ROUTES, temporal=TEMPORAL)


def slope_proxy(voltage, name, noise_rng):
    """Detached parent-state proxy; noise is independent of minibatch sampling."""
    base = name.removeprefix('shuffle_').removeprefix('mean_')
    if base.startswith('noise'):
        sd = {'noise025':.25,'noise05':.5,'noise1':1.}[base]
        voltage = voltage + sd*torch.randn(voltage.shape, dtype=voltage.dtype, generator=noise_rng)
        return 1-voltage.tanh().square()
    if base in {'bins2','bins4'}:
        # Positive shunting conductances bound these parent voltages by 1.
        # Use the physical range, not bins outside the reachable state space.
        edges = voltage.new_tensor([.5] if base=='bins2' else [.25,.5,.75])
        centers = voltage.new_tensor([.25,.75] if base=='bins2' else [.125,.375,.625,.875])
        return (1-centers.tanh().square())[torch.bucketize(voltage.abs().contiguous(),edges)]
    if base=='derivative': return 1-voltage.tanh().square()
    raise ValueError(name)


def context_mean(values, contexts):
    result = values.clone()
    for c in torch.unique(contexts):
        ix = contexts==c
        result[ix] = values[ix].mean(0)
    return result


def memory_state(sequence, raw_decay, log_gain, mode):
    """Eight independent leaky recurrent traces with exact online eligibilities.

The one-step control keeps the same forward memory but drops accumulated
parameter eligibility. The BPTT mode is a verification reference only.
"""
    alpha, gain = raw_decay.sigmoid(), log_gain.exp()
    if mode=='bptt':
        state = torch.zeros_like(sequence[:,0])
        for x in sequence.unbind(1): state = alpha*state+(1-alpha)*gain*x
        return 3*state
    with torch.no_grad():
        if mode=='none': alpha = torch.zeros_like(alpha)
        state = torch.zeros_like(sequence[:,0]); ea = state.clone(); eg = state.clone()
        for x in sequence.unbind(1):
            keep = alpha if mode=='trace' else 0.
            ea = keep*ea + alpha*(1-alpha)*(state-gain*x)
            eg = keep*eg + (1-alpha)*gain*x
            state = alpha*state+(1-alpha)*gain*x
        if mode=='none': ea.zero_(); eg.zero_()
    return 3*(state + (raw_decay-raw_decay.detach())*ea + (log_gain-log_gain.detach())*eg)


def temporal_dataset(seed, split, size, length=8):
    offsets = dict(train=1,validation=2,test=3,diagnostic=4,ood=5)
    rng = np.random.default_rng(np.random.SeedSequence([seed,offsets[split],length,739]))
    sequence = torch.tensor(rng.uniform(-2,2,(size,length,8)),dtype=torch.float64)
    cue = torch.tensor(np.eye(4)[rng.integers(4,size=size)],dtype=torch.float64)
    state = torch.zeros(size,8,dtype=torch.float64)
    for x in sequence.unbind(1): state = .8*state+.2*x
    features = (3*state).reshape(-1,4,2).tanh()
    evidence = .5*features.sum(-1)+.25*features.prod(-1)
    y = (cue*evidence*torch.tensor([1.,-1.,1.,-1.])).sum(-1)
    return sequence,cue,y


class ExtensionNet(RescueNet):
    def __init__(self, seed, study='proxy', arm='derivative'):
        super().__init__(seed)
        if arm not in ARMS[study]: raise ValueError((study,arm))
        self.study, self.arm = study, arm
        self.noise_rng = torch.Generator().manual_seed(seed+104729)
        self.randomize_rng = torch.Generator().manual_seed(seed+130363)
        self.contexts = None
        if study=='routing':
            rng = torch.Generator().manual_seed(seed+15485863)
            self.route_logits = nn.Parameter(.02*torch.randn(4,4,generator=rng),
                                             requires_grad=arm.startswith('learned_'))
        if study=='temporal':
            self.raw_decay = nn.Parameter(torch.zeros(8),requires_grad=arm!='exact_no_memory')
            self.log_gain = nn.Parameter(torch.zeros(8),requires_grad=arm!='exact_no_memory')

    def route_probabilities(self, cue):
        if self.arm=='oracle_augmented': return cue
        if self.arm=='uniform_augmented': return torch.ones_like(cue)/4
        if self.arm=='wrong_augmented': return cue.roll(1,-1)
        return (cue@self.route_logits).softmax(-1)

    def forward(self, excitation, inhibition):
        if self.study=='temporal':
            mode = 'none' if self.arm=='exact_no_memory' else ('one_step' if self.arm.endswith('one_step') else 'trace')
            state = memory_state(excitation,self.raw_decay,self.log_gain,mode)
            cue = inhibition
            sensory = torch.cat([state.exp(),(-state).exp()],-1)
            excitation = torch.cat([sensory,cue],-1)
            inhibition = torch.cat([sensory,4*(1-cue)],-1)
        cue = excitation[:,-4:]
        self.contexts = cue.argmax(-1)
        if self.study=='routing':
            # Fixed total inhibitory activity 12; cue reaches only the router.
            inhibition = torch.cat([inhibition[:,:-4],4*(1-self.route_probabilities(cue))],-1)
            excitation = torch.cat([excitation[:,:-4],torch.zeros_like(cue)],-1)
        return super().forward(excitation,inhibition)

    def learning_rule(self):
        if self.study=='proxy': return self.arm
        if self.arm.startswith('exact'): return 'exact'
        if 'resistance' in self.arm: return 'resistance'
        return 'derivative'

    @torch.no_grad()
    def terminal_gain(self, rule):
        h = self.distal_gate('resistance')
        if rule=='resistance': return h
        v = self.core.branch_layers[1]._last_branch_diagnostics['V']
        sensitivity = slope_proxy(v,rule,self.noise_rng)
        if rule.startswith('shuffle_'):
            sensitivity = context_shuffle(sensitivity,self.contexts,self.randomize_rng)
        if rule.startswith('mean_'): sensitivity = context_mean(sensitivity,self.contexts)
        return h*sensitivity.repeat_interleave(2,-1)

    def gradients(self, excitation, inhibition, target, variance, rule=None):
        rule = self.learning_rule() if rule is None else rule
        self.zero_grad(set_to_none=True)
        router_gradient = None
        if self.study=='routing' and self.arm=='learned_exact_router_augmented':
            self.local = False
            prediction = self(excitation,inhibition)
            loss = .5*(prediction-target).square().mean()/variance
            router_gradient = torch.autograd.grad(loss,self.route_logits)[0].detach()
        self.local = rule!='exact'
        prediction = self(excitation,inhibition)
        loss = .5*(prediction-target).square().mean()/variance
        objective = loss
        if self.local:
            error = torch.autograd.grad(loss,self.outputs[-1],retain_graph=True)[0].detach()
            gain = self.terminal_gain(rule)
            for index,output in enumerate(self.outputs[:-1]):
                delivered = error.repeat_interleave(output.shape[1]//self.somata,-1)
                if index==0: delivered = delivered*gain
                objective = objective + (output*delivered).sum()
        objective.backward()
        if router_gradient is not None: self.route_logits.grad = router_gradient
        self.local = False
        return float(loss.detach())*2


def mechanistic_diagnostic(net, data, variance):
    """Target-interaction contribution at one state, not a training intervention."""
    x,i,y = data
    if net.study=='temporal': return []
    features=x[:,:8].log().reshape(-1,4,2).tanh()
    separable=(x[:,-4:]*.5*features.sum(-1)*x.new_tensor([1.,-1.,1.,-1.])).sum(-1)
    rng=(net.noise_rng.get_state(),net.randomize_rng.get_state())
    params=list(net.core.branch_layers[0].parameters()); rows=[]; vectors={}
    for rule in list(dict.fromkeys(['exact','resistance',net.learning_rule()])):
        pair=[]
        for target in [y,separable]:
            net.noise_rng.set_state(rng[0]);net.randomize_rng.set_state(rng[1])
            net.gradients(x,i,target,variance,rule)
            pair.append(torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten() for p in params]).detach().clone())
        vectors[rule]=(pair[0],pair[0]-pair[1])
    for rule,(v,inter) in vectors.items():
        ref,ri=vectors['exact']
        rows.append(dict(rule=rule,terminal_cosine=float(v@ref/(v.norm()*ref.norm()).clamp_min(1e-30)),
            interaction_projection=float(inter@ri/ri.square().sum().clamp_min(1e-30)),
            interaction_cosine=float(inter@ri/(inter.norm()*ri.norm()).clamp_min(1e-30))))
    net.noise_rng.set_state(rng[0]);net.randomize_rng.set_state(rng[1])
    return rows


def train(seed, study, arm, rate, steps=4096, phase='development'):
    started=time.monotonic()
    datafn=temporal_dataset if study=='temporal' else dataset
    training=datafn(seed,'train',2048); validation=datafn(seed,'validation',1024)
    # The frozen selection default is separable; these extensions use interaction.
    if study!='temporal':
        training=dataset(seed,'train',2048,'interaction');validation=dataset(seed,'validation',1024,'interaction')
    net=ExtensionNet(seed,study,arm).double()
    variance=training[2].var(unbiased=False)
    opt=torch.optim.Adam(net.parameters(),lr=rate)
    minibatches=torch.Generator().manual_seed(seed+100)
    best=evaluate(net,validation); best_state=copy.deepcopy(net.state_dict());best_step=0
    history=[dict(step=0,validation_nmse=best,bound_fraction=bound_fraction(net,9))]
    bounds=clips=0; active=active_values(net)
    for step in range(1,steps+1):
        ix=torch.randint(2048,(128,),generator=minibatches)
        net.gradients(training[0][ix],training[1][ix],training[2][ix],variance)
        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10.)
        if not torch.isfinite(norm):raise FloatingPointError((seed,study,arm,rate,step))
        clips+=int(norm>10);opt.step()
        with torch.no_grad():
            hit=False
            for p,mask in active:
                values=p[mask] if mask is not None else p
                hit |= bool((values.abs()>9).any());p.clamp_(-9,9)
            if study=='routing':net.route_logits.clamp_(-9,9)
            if study=='temporal':net.raw_decay.clamp_(-9,9);net.log_gain.clamp_(-3,3)
        bounds+=int(hit)
        if step%128==0 or step==steps:
            value=evaluate(net,validation)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite validation loss')
            history.append(dict(step=step,validation_nmse=value,bound_fraction=bound_fraction(net,9)))
            if value<best:best,best_step,best_state=value,step,copy.deepcopy(net.state_dict())
    endpoint=copy.deepcopy(net.state_dict());net.load_state_dict(best_state)
    result=dict(seed=seed,study=study,arm=arm,rate=rate,steps=steps,phase=phase,
                validation_nmse=best,selected_step=best_step,bounds=bounds,clips=clips,
                selected_bound_fraction=bound_fraction(net,9),history=history)
    if study=='routing':
        with torch.no_grad():probs=net.route_probabilities(torch.eye(4,dtype=torch.float64))
        result.update(route_probabilities=probs.tolist(),routing_accuracy=float((probs.argmax(-1)==torch.arange(4)).double().mean()),
                      correct_route_mass=float(probs.diag().mean()))
    if study=='temporal':result.update(memory_decay=net.raw_decay.sigmoid().tolist(),memory_gain=net.log_gain.exp().tolist())
    if phase=='fresh':
        if study=='temporal':
            result['test_nmse']=evaluate(net,temporal_dataset(seed,'test',4096))
            result['length_nmse']={str(t):evaluate(net,temporal_dataset(seed,'ood',4096,t)) for t in [4,8,16]}
        else:
            result['test_nmse']=evaluate(net,dataset(seed,'test',4096,'interaction'))
            result['stress3_nmse']=evaluate(net,dataset(seed,'ood',4096,'interaction',3.))
            result['diagnostic']=mechanistic_diagnostic(net,dataset(seed,'diagnostic',512,'interaction'),variance)
    result['elapsed_seconds']=time.monotonic()-started
    return result,dict(selected=best_state,endpoint=endpoint)
