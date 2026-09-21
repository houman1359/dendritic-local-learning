"""Describe learned computations at existing validation-selected checkpoints.

No optimization, seed selection or model selection is performed here. The
population uses verified execution code; the single-neuron code and states
are checked against their original execution records before evaluation.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def interaction_component(surface, weights):
    """Product-measure ANOVA interaction; axis 0=z1, axis 1=z2."""
    return surface-(surface@weights)[:,None]-(weights@surface)[None,:]+weights@surface@weights


def main(args):
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true')
    sys.dont_write_bytecode=True
    J=args.journal.resolve();args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post hoc checkpoint-only visualization; all retained primary seeds, no training or outcome-based selection',
        single_neuron='20 opposed-target seeds; primary Adam rate 0.03; validation-selected within 4096 updates; exact, hard distal gate, broadcast and each seed teacher',
        tuning='Selected proximal branch contribution to somatic voltage while z1 varies; integrate independent z2 with 32-point Gauss-Legendre quadrature on [-2,2]',
        population='20 original-bound Adam rescue seeds; selected rates; exact, resistance and augmented rules',
        surface='25 by 25 uniform grid on [-2,2]^2, each cue; average 16 fixed antithetic irrelevant-stream draws shared across seeds and rules',
        context_alignment='Multiply each predicted surface by (-1)^cue, then average four cues within each seed',
        interaction='Subtract each one-feature marginal mean and add grand mean, using normalized trapezoidal product weights on the grid',
        uncertainty='Pointwise 95% whole-seed bootstrap intervals for tuning; surfaces show all-seed means; per-seed component errors are descriptive',
        source_sha256=sha(__file__))
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    inputs={}; single_rows=[]
    root=J/'source_data/conductance_local_gate'
    freeze=json.loads((root/'protocol_freeze.json').read_text())
    for relative,expected in freeze['scientific_source_sha256'].items():
        assert sha(J/relative)==expected,relative
    config=json.loads((root/'protocol.json').read_text())
    assert sha(root/'protocol.json')==freeze['protocol_sha256']
    model=load_module('verified_single_neuron',J/'scripts/conductance_local_gate/model.py')
    grid=np.linspace(-2,2,65);nuisance,w=np.polynomial.legendre.leggauss(32);nuisance*=2;w/=2
    for seed in config['fresh_seeds']:
        audit=json.loads((root/'runs'/f'seed_{seed}_audit.json').read_text())
        stem=f'seed_{seed}_opposed_strong'
        for suffix in ['_states.npz','_config.json']:
            path=root/'runs'/(stem+suffix);assert sha(path)==audit['files_sha256'][path.name];inputs[str(path)]=sha(path)
        archive=np.load(root/'runs'/(stem+'_states.npz'))
        meta=json.loads((root/'runs'/(stem+'_config.json')).read_text())['task_metadata']
        window=list(archive['best_windows']).index(4096)
        rules=['unit_broadcast','hard_distal_unit_proximal','exact']
        states=[]
        for rule in rules:
            ix=next(i for i,r in enumerate(meta['records']) if r['rule']==rule and r['rate']==.03)
            states.append(archive['best_theta'][window,ix])
        states.append(np.asarray(meta['teacher_log_conductances']));rules.append('target')
        for context in [0,1]:
            z1,z2=np.meshgrid(grid,nuisance,indexing='ij');z=np.column_stack([z1.ravel(),z2.ravel()]);xx=np.tile(z,(1,2))
            x=np.column_stack([np.exp(xx),np.exp(-xx),np.full(len(z),10*(context==1)),np.full(len(z),10*(context==0))])
            f=model.forward(np.asarray(states),x);g=f['conductance'];vp=f['voltage'][:,:,4+context]
            contribution=vp*g[:,22+context,None]/(1+g[:,22:24].sum(1))[:,None]
            contribution=contribution.reshape(4,len(grid),len(nuisance))@w
            output=f['output'].reshape(4,len(grid),len(nuisance))@w
            for i,rule in enumerate(rules):
                for k,feature in enumerate(grid):single_rows.append(dict(seed=seed,context=context,rule=rule,z1=feature,branch_contribution=contribution[i,k],somatic_output=output[i,k]))
    pd.DataFrame(single_rows).to_csv(args.output/'branch_tuning.csv',index=False)
    print('Single-neuron tuning complete',flush=True)

    launcher=load_module('verified_population_launcher',J/'code/population_replay/launch.py')
    frozen,identity,original=launcher.verified_sources()
    sys.path[:0]=[str(frozen/d) for d in ('study','selection','base','runtime/src')]
    import torch
    from rescue import RescueNet,dataset
    import dendritic_modeling
    assert Path(dendritic_modeling.__file__).resolve().is_relative_to(frozen/'runtime')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    grid=np.linspace(-2,2,25);z1,z2=np.meshgrid(grid,grid,indexing='ij')
    weights=np.ones(len(grid));weights[[0,-1]]=.5;weights/=weights.sum()
    target=.5*(np.tanh(z1)+np.tanh(z2))+.25*np.tanh(z1)*np.tanh(z2)
    target_interaction=interaction_component(target,weights)
    assert np.max(np.abs(target_interaction-.25*np.tanh(z1)*np.tanh(z2)))<1e-14
    # Fixed, paired marginalization samples; symmetric pairs remove an avoidable
    # finite-sample mean without changing the uniform input distribution.
    rng=np.random.default_rng(2026092117);base=rng.uniform(-2,2,(8,4,2));background=np.concatenate([base,-base])
    batches=[]
    for context in range(4):
        latent=np.broadcast_to(background,(grid.size**2,16,4,2)).copy()
        latent[:,:,context,0]=z1.reshape(-1,1);latent[:,:,context,1]=z2.reshape(-1,1)
        latent=latent.reshape(-1,8);cue=np.eye(4)[np.full(len(latent),context)]
        sensory=np.concatenate([np.exp(latent),np.exp(-latent)],axis=1)
        batches.append((torch.tensor(np.concatenate([sensory,cue],axis=1)),torch.tensor(np.concatenate([sensory,4*(1-cue)],axis=1))))
    rows=[];metrics=[]
    for rule in ['resistance','derivative','exact']:
        results=sorted((args.population_archive/'fresh/results').glob(f'*_adam_b9_{rule}_r*.json'))
        assert len(results)==20,(rule,len(results))
        for file in results:
            record=json.loads(file.read_text());seed=record['seed']
            path=args.population_archive/'fresh/checkpoints'/(file.stem+'.pt')
            assert sha(path)==record['checkpoint_sha256'];inputs[str(file)]=sha(file);inputs[str(path)]=sha(path)
            net=RescueNet(seed).double();net.calibrate(*dataset(seed,'train',2048,'interaction')[:2])
            net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['selected'])
            contexts=[]
            with torch.no_grad():
                for c,(x,inh) in enumerate(batches):
                    pred=torch.cat([net(x[k:k+1024],inh[k:k+1024]).reshape(-1) for k in range(0,len(x),1024)]).numpy()
                    surface=((-1)**c)*pred.reshape(grid.size**2,16).mean(1).reshape(z1.shape)
                    contexts.append(surface)
                    for k in range(len(grid)):
                        for l in range(len(grid)):rows.append(dict(seed=seed,rule=rule,context=c,z1=grid[k],z2=grid[l],prediction=surface[k,l],target=target[k,l]))
            average=np.mean(contexts,axis=0);component=interaction_component(average,weights)
            for component_name,actual,expected in [('full',average,target),('interaction',component,target_interaction),('additive',average-component,target-target_interaction)]:
                mse=float(weights@((actual-expected)**2)@weights)
                energy=float(weights@(expected**2)@weights)
                metrics.append(dict(seed=seed,rule=rule,component=component_name,mse=mse,target_energy=energy,relative_mse=mse/energy))
            print(rule,seed,'complete',flush=True)
    pd.DataFrame(rows).to_csv(args.output/'population_surfaces.csv',index=False)
    pd.DataFrame(metrics).to_csv(args.output/'component_errors.csv',index=False)
    report=dict(protocol=protocol,inputs=inputs,population_runtime_identity_sha256=sha(frozen/'identity.json'),
        outputs={p.name:sha(p) for p in args.output.glob('*.csv')},torch=torch.__version__,numpy=np.__version__,
        complete=True,training_runs=0,single_seeds=20,population_seeds=20)
    (args.output/'checkpoint_computation_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(metrics).groupby(['rule','component']).relative_mse.mean().to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--journal',type=Path,required=True);p.add_argument('--population-archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
