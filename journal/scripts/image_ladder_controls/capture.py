#!/usr/bin/env python3
"""Activation/voltage coordinate audit at independent MNIST checkpoints.

Both measures are retained: mean per-example-neuron captured-energy ratios and
ratio of total captured energy. The tested delivery dictionary acts in
activation coordinates; the historical atlas acts in voltage coordinates.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from run import OUT, RUNTIME, PROJECT, OLD, sha, dump, tensor_hash
sys.path.insert(0,str(RUNTIME/'src'))
import torch
from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

def probe():
    root=PROJECT/'data/mnist/MNIST/raw'
    x=np.fromfile(root/'t10k-images-idx3-ubyte',dtype=np.uint8)[16:].reshape(-1,784)[:2048]
    y=np.fromfile(root/'t10k-labels-idx1-ubyte',dtype=np.uint8)[8:][:2048]
    return torch.tensor(x,dtype=torch.float32)/255,torch.tensor(y,dtype=torch.long)

def measure(config_path,checkpoint,device='cpu'):
    cfg=load_config(str(config_path));torch.manual_seed(int(cfg.experiment.model_seed))
    model,_=initialize_model(cfg.model)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(state,strict=True);model=model.to(device).eval()
    cells=model.core_network.layers[0].excitatory_cells;layers=list(cells.branch_layers[:3]);assert cells.n_soma==128
    x,y=probe();metrics={};forward_difference=0.;captures={}
    for layer in layers:layer._store_analysis_currents=True
    for coordinate in ('activation','voltage'):
        rows=[]
        for start in range(0,len(x),256):
            caught={};handles=[]
            if coordinate=='activation':
                for i,layer in enumerate(layers):
                    handles.append(layer.register_forward_hook(lambda module,args,output,i=i:caught.__setitem__(i,output)))
            xb=x[start:start+256].to(device);yb=y[start:start+256].to(device)
            with torch.no_grad():reference=model(xb)
            caught.clear();logits=model(xb)
            forward_difference=max(forward_difference,float((logits.detach()-reference).abs().max()))
            for h in handles:h.remove()
            if coordinate=='voltage':
                for i,layer in enumerate(layers):caught[i]=layer._last_analysis_currents['pre_gate_voltage']
            loss=torch.nn.functional.cross_entropy(logits,yb,reduction='sum')
            grad=torch.autograd.grad(loss,[caught[0],caught[1],caught[2]])
            fields=torch.cat([grad[1].reshape(-1,128,3),grad[0].reshape(-1,128,9)],-1).detach().double().cpu().reshape(-1,12)
            energy=fields.square().sum(1);keep=energy>0;fields=fields[keep];energy=energy[keep]
            coeff=(fields[:,:3]+fields[:,3:].reshape(-1,3,3).sum(-1))/4
            for name,captured in [('broadcast_k1',12*fields.mean(1).square()),('subtrees_k3',4*coeff.square().sum(1)),('exact_k12',energy)]:
                ratios=captured/energy
                rows.append(dict(basis=name,n_fields=len(energy),sum_ratio=float(ratios.sum()),captured_energy=float(captured.sum()),total_energy=float(energy.sum())))
        df=pd.DataFrame(rows).groupby('basis',as_index=False).sum()
        for row in df.to_dict('records'):
            captures[(coordinate,row['basis'])]=dict(coordinate=coordinate,basis=row['basis'],n_fields=int(row['n_fields']),mean_capture=row['sum_ratio']/row['n_fields'],ensemble_energy_capture=row['captured_energy']/row['total_energy'])
    assert forward_difference<=1e-6
    for coordinate in ('activation','voltage'):
        assert captures[(coordinate,'subtrees_k3')]['mean_capture']>=captures[(coordinate,'broadcast_k1')]['mean_capture']-1e-12
    return list(captures.values()),forward_difference

def run(mode,index,device):
    torch.set_num_threads(1)
    if mode=='historical':
        a=('shunting','additive')[index//15];i=index%15;stamp='20260825173517' if a=='shunting' else '20260825173519'
        root=OLD/f'journal_mnist_feedback_ladder_exact_path_{a}_15seed_{stamp}'/f'results/config_{i}'
        cfg=root/'config.json';checkpoints=[('trained',root/'final_model.pt')];seed=42+i
    else:
        selection=pd.DataFrame(json.loads((OUT/'selection.json').read_text())['selected'])
        conditions=pd.DataFrame(json.loads((OUT/'fresh_conditions.json').read_text()))
        selected=conditions[conditions.arm.eq('exact_path')].merge(selection[['architecture','arm','multiplier']],on=['architecture','arm','multiplier']).sort_values(['architecture','seed']).reset_index(drop=True)
        record=selected.iloc[index];a=record.architecture;seed=int(record.seed);root=Path(record.results_dir)
        audit=json.loads((root/'run_audit.json').read_text());cfg=Path(audit['model_results_dir'])/'config.json'
        checkpoints=[('initial',root/'initial_model.pt'),('trained',Path(audit['model_results_dir'])/'final_model.pt')]
    rows=[]
    for state,path in checkpoints:
        data,difference=measure(cfg,path,device=device)
        rows.extend(dict(cohort=mode,architecture=a,seed=seed,checkpoint=state,checkpoint_sha256=sha(path),forward_max_difference=difference,**r) for r in data)
    destination=OUT/'capture'/f'{mode}_{index:03d}.csv';destination.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(destination,index=False)
    print(destination,flush=True)

def summarize():
    files=sorted((OUT/'capture').glob('fresh_*.csv'));assert len(files)==20
    df=pd.concat([pd.read_csv(p) for p in files]);df.to_csv(OUT/'summaries/delivery_coordinate_capture.csv',index=False)
    from analyze import interval
    rows=[]
    for key,g in df.groupby(['architecture','checkpoint','coordinate','basis']):
        for metric in ('mean_capture','ensemble_energy_capture'):
            rows.append(dict(zip(('architecture','checkpoint','coordinate','basis'),key),metric=metric,**interval(g[metric])))
    pd.DataFrame(rows).to_csv(OUT/'summaries/delivery_coordinate_capture_summary.csv',index=False)
    dump(OUT/'summaries/capture_audit.json',dict(checkpoints=40,probe_examples=2048,neurons_per_example=128,soma_excluded=True,
        activation='Derivative w.r.t.compartment output (delivered learning signal); activation derivative remains localeligibility',
        voltage='Derivative w.r.t.pre-reactivation voltage (historicalatlas definition)',
        basis='Three proximal-plus3distalchild indicator columns; exactidentity12; uniformbroadcast1',
        zero_fields='Drop exactlyzeroenergy fields, retaincounts',field_weighting='mean_capture:equalnonzerofieldratios; ensemble_energy_capture:totalenergyratio',
        protocol_sha256=sha(OUT/'protocol.json'),source_sha256=sha(Path(__file__))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['historical','fresh','summarize'],default='fresh');p.add_argument('--index',type=int,default=0);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.mode=='summarize':summarize()
    else:run(a.mode,a.index,a.device)
