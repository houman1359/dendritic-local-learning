"""Describe proxy fidelity and occupied voltage bins at retained fresh states.

This is a post hoc mechanistic description. It does not select models, alter
the prospective endpoints, or constitute an additional training cohort.
"""
from pathlib import Path
import argparse,importlib.util,json,hashlib,sys
import numpy as np
import pandas as pd
J=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('extension_diagnostics',J/'code/optional_extensions/experiments.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def main(args):
    e.torch.set_num_threads(1);e.torch.set_num_interop_threads(1)
    protocol=json.loads((args.root/'fresh_protocol.json').read_text());rows=[];inputs={};router_rows=[]
    for job in protocol['jobs']:
        if job['study']!='proxy':continue
        seed,arm,rate=job['seed'],job['arm'],job['rate']
        key=f'proxy_{arm}_s{seed}_r{rate:g}'
        result=json.loads((args.root/'fresh/results'/(key+'.json')).read_text())
        checkpoint=args.root/'fresh/checkpoints'/(key+'.pt')
        assert e.sha(checkpoint)==result['checkpoint_sha256'];inputs[key]=e.sha(checkpoint)
        net=e.ExtensionNet(seed,'proxy',arm).double()
        net.load_state_dict(e.torch.load(checkpoint,map_location='cpu',weights_only=True)['selected'])
        x,i,y=e.dataset(seed,'diagnostic',512,'interaction')
        with e.torch.no_grad():
            net(x,i);v=net.core.branch_layers[1]._last_branch_diagnostics['V']
            true=1-v.tanh().square();context=x[:,-4:].argmax(-1)
            group=(e.torch.arange(v.shape[1])[None,:]%4)==context[:,None]
            if arm in {'exact','derivative'}:q=true.clone()
            elif arm=='resistance':q=e.torch.ones_like(true)
            else:
                q=e.slope_proxy(v,arm,e.torch.Generator().manual_seed(seed+32452843))
                if arm.startswith('shuffle_'):q=e.context_shuffle(q,context,e.torch.Generator().manual_seed(seed+49979687))
                if arm.startswith('mean_'):q=e.context_mean(q,context)
            # Remove parent/context means before assessing example dependence.
            qt=q-e.context_mean(q,context);tt=true-e.context_mean(true,context)
            for name,mask in [('selected',group),('unselected',~group)]:
                a,b=qt[mask],tt[mask];denom=float(a.norm()*b.norm())
                corr=float(a@b/denom) if denom>1e-14 else None
                common=dict(seed=seed,arm=arm,rate=rate,parents=name,voltage_min=float(v[mask].min()),voltage_max=float(v[mask].max()),
                            sensitivity_mse=float((q[mask]-true[mask]).square().mean()),conditional_correlation=corr,
                            conditional_proxy_sd=float(a.square().mean().sqrt()),
                            occupied_proxy_values=int(q[mask].unique().numel()) if 'bins' in arm and not arm.startswith('mean') else None)
                for band,(lo,hi) in enumerate(zip([0,.25,.5,.75],[.25,.5,.75,1.])):
                    common[f'voltage_band_{band}_fraction']=float(((v[mask]>=lo)&(v[mask]<hi)).double().mean())
                for policy in ['common','selected']:
                    if rate==(.03 if policy=='common' else protocol['selection']['proxy'][arm]['rate']):rows.append({**common,'policy':policy})
    args.output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(args.output/'proxy_fidelity.csv',index=False)
    for job in protocol['jobs']:
        if job['study']!='routing':continue
        seed,arm,rate=job['seed'],job['arm'],job['rate'];key=f'routing_{arm}_s{seed}_r{rate:g}'
        result=json.loads((args.root/'fresh/results'/(key+'.json')).read_text())
        checkpoint=args.root/'fresh/checkpoints'/(key+'.pt')
        assert e.sha(checkpoint)==result['checkpoint_sha256'];inputs[key]=e.sha(checkpoint)
        state=e.torch.load(checkpoint,map_location='cpu',weights_only=True)['selected'];logits=state['route_logits']
        for policy in ['common','selected']:
            if rate!=(.03 if policy=='common' else protocol['selection']['routing'][arm]['rate']):continue
            router_rows.append(dict(seed=seed,arm=arm,rate=rate,policy=policy,trainable_router=arm.startswith('learned_'),
                selected_logit_abs_max=float(logits.abs().max()),selected_logit_bound_fraction=float((logits.abs()>=9-1e-10).double().mean()),
                matching_stream_probability=result['correct_route_mass']))
    pd.DataFrame(router_rows).to_csv(args.output/'router_selected_state.csv',index=False)
    (args.output/'provenance.json').write_text(json.dumps(dict(scope=__doc__,source_sha256=e.sha(__file__),input_checkpoints=inputs,
        outputs={p.name:e.sha(p) for p in args.output.glob('*.csv')},
        controller_bounds='Selected-state occupancy only; the training log counts core-conductance contacts, not transient controller-logit contacts.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
