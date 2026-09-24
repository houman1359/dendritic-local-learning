"""Audit the uniform-device stopping extension and export paired trajectories.

All original conditions and seeds are retained. Curves use validation-best
states and carry those states forward after ordinary early stopping.
"""
from pathlib import Path
import argparse, copy, hashlib, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch, yaml

CONTRASTS = [
    ('depth_gain_exact_bp', ('exact_autograd_bp_recipe',3), ('exact_autograd_bp_recipe',1)),
    ('localca_path_minus_shared', ('path_transport',3), ('per_soma_shared',3)),
    ('bp_exact_minus_broadcast', ('exact_autograd_bp_recipe',3), ('broadcast_autograd_bp_recipe',3)),
    ('broadcast_bp_minus_localca_recipe', ('broadcast_autograd_bp_recipe',3), ('broadcast_autograd_localca_recipe',3)),
]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, d):
    Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def bounds(values, weights):
    values=np.asarray(values,float)
    b=weights@values
    lo,hi=np.quantile(b,[.025,.975],axis=0)
    return values.mean(axis=0),lo,hi

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-run',type=int,help='Audit one completed run without producing cohort estimates')
    args=parser.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text())
    records=protocol['retained']+protocol['records']
    assert len(records)==60 and len({r['index'] for r in records})==60
    if args.check_run is not None:
        records=[r for r in records if r['index']==args.check_run]
        assert len(records)==1
    histories=[];checks=[];runs=[];flags=[]
    for rec in sorted(records,key=lambda r:r['index']):
        folder=Path(rec['result_dir']); receipt=json.loads((folder/'execution.json').read_text())
        assert receipt['status']=='complete' and not receipt['smoke']
        if 'exit_code' in receipt:
            assert receipt['exit_code']==0
        else:
            # The original completed-run driver predates the repair receipt's
            # exit_code field. It records completion only after the checkpoint
            # and metric checks; accept this schema only for retained runs.
            assert rec['index'] in {r['index'] for r in protocol['retained']}
            assert receipt['patience_stopped'] is True and receipt['reached_cap'] is False
        assert receipt['finished_unix'] > receipt['started_unix']
        assert 'H200' in receipt['gpu']
        assert receipt.get('runtime_commit',receipt.get('source',{}).get('commit'))==protocol['runtime_commit']
        assert sha(rec['config'])==rec['config_sha256']
        original=yaml.safe_load(Path(rec['original_config']).read_text())
        cfgfile=folder/'executed_config.yaml'
        if not cfgfile.exists():cfgfile=Path(rec['config'])
        cfg=yaml.safe_load(cfgfile.read_text());normalized=copy.deepcopy(cfg)
        normalized['outputs']=original['outputs']
        normalized['training']['main']['common']['epochs']=original['training']['main']['common']['epochs']
        assert normalized==original, rec['index']
        assert sha(cfgfile)==receipt['config_sha256']
        cap=cfg['training']['main']['common']['epochs']
        summary=json.loads((folder/'training_summary.json').read_text())
        losses=np.asarray(summary['valid_losses'],float);train=np.asarray(summary['train_losses'],float)
        n=len(losses);assert n==len(train)>0 and np.isfinite(losses).all() and np.isfinite(train).all()
        assert receipt['epochs']==n and receipt['best_epoch']==summary['best_epoch']
        assert int(summary['best_epoch'])==int(losses.argmin())+1
        assert abs(summary['best_loss']-losses.min())<1e-6
        statepath=folder/f'state_{n}.pt'; state=torch.load(statepath,map_location='cpu',weights_only=False)
        final=torch.load(folder/'final_model.pt',map_location='cpu',weights_only=False)
        assert final.keys()==state['best_model'].keys()
        assert all(torch.equal(final[k],state['best_model'][k]) for k in final)
        assert state['loss_lists'][1]==summary['valid_losses']
        assert set(state['loader_generators'])=={'train_loader','valid_loader'}
        assert state['source_commit']==protocol['runtime_commit']
        resources=json.loads((folder/'model_resources.json').read_text())
        assert resources['trainable_parameters']==66178 and resources['active_synapses']==14336
        stopped=n<cap and state['patience_counter']>=30
        meta={k:rec[k] for k in ['index','arm','depth','seed']}
        if not stopped:flags.append(meta|dict(epochs=n,cap=cap,best_epoch=int(summary['best_epoch'])))
        best_indices=np.empty(n,dtype=int);best=0
        for k in range(n):
            if losses[k]<losses[best]:best=k
            best_indices[k]=best+1
        histories.extend(meta|dict(epoch=k+1,validation_loss=float(v),train_loss=float(t),
            best_epoch=int(best_indices[k]),best_loss=float(losses[best_indices[k]-1]))
            for k,(v,t) in enumerate(zip(losses,train)))
        restored=receipt.get('restored_epoch',0)
        if restored:
            assert sha(rec['resume_checkpoint'])==rec['resume_sha256']==receipt['resume_sha256']
            before=torch.load(rec['resume_checkpoint'],map_location='cpu',weights_only=False)
            assert before['epoch']==restored and before['loss_lists'][1]==summary['valid_losses'][:restored]
        checks.append(meta|dict(epochs=n,best_epoch=int(summary['best_epoch']),cap=cap,
            ordinary_stopping_reached=stopped,restored_epoch=restored,
            final_equals_best=True,private_loader_rng_saved=True,
            checkpoint_sha256=sha(statepath),execution_sha256=sha(folder/'execution.json'),
            config_sha256=sha(cfgfile),training_summary_sha256=sha(folder/'training_summary.json')))
        runs.append((rec,meta,folder,losses,best_indices,n,restored))
        print('Audited',rec['index'],n,flush=True)
    assert not flags, f'Unfinished stopping criteria: {flags}'
    if args.check_run is not None:
        dump(out/f'run_{args.check_run}_audit.json',checks[0])
        return
    maxepoch=max(x[5] for x in runs)
    # A fixed display rule, independent of performance: retain every epoch to
    # 600, then 256 logarithmically spaced epochs plus budgets and stop times.
    epochs=sorted(set(range(1,601)) | set(np.rint(np.geomspace(601,maxepoch,256)).astype(int))
                  | {180,600,6000,maxepoch} | {x[5] for x in runs})
    epochs=[e for e in epochs if e<=maxepoch]
    trajectories=[];endpoints=[];metric_audit=[]
    for rec,meta,folder,losses,best_indices,n,restored in runs:
        # Selected epochs depend only on validation losses. Read immutable
        # evaluation JSONs concurrently, then assemble in chronological order.
        # Each file is read once; its hash authenticates those exact bytes.
        selected_epochs=sorted({int(best_indices[min(e,n)-1]) for e in epochs})
        def load_selected(selected):
            source=Path(rec['prior_results']) if restored and selected<=restored else folder
            p=source/'performance/epochs'/f'epoch{selected}.json'
            raw=p.read_bytes();d=json.loads(raw)
            assert abs(-d['categorical_loglikelihood']['valid']-losses[selected-1])<2e-6
            assert all(np.isfinite(d[q][split]) for q in ['accuracy','categorical_loglikelihood'] for split in ['train','valid','test'])
            return selected,d,meta|dict(selected_epoch=selected,source=str(p),sha256=hashlib.sha256(raw).hexdigest(), **{f'{quantity}_{split}':float(d[quantity][split]) for quantity in ['accuracy','categorical_loglikelihood'] for split in ['train','valid','test']})
        with ThreadPoolExecutor(max_workers=8) as readers:
            loaded=list(readers.map(load_selected,selected_epochs))
        selected_cache={selected:d for selected,d,_ in loaded}
        metric_audit.extend(audit for _,_,audit in loaded)
        for epoch in epochs:
            observed=min(epoch,n);selected=int(best_indices[observed-1])
            d=selected_cache[selected]
            trajectories.append(meta|dict(epoch=int(epoch),selected_epoch=selected,
                observed_epoch=epoch<=n,last_training_epoch=n,
                test_accuracy=float(d['accuracy']['test']),
                test_cross_entropy=float(-d['categorical_loglikelihood']['test']),
                best_validation_loss=float(losses[selected-1])))
        for budget in [180,600,6000,maxepoch]:
            q=next(r for r in reversed(trajectories) if r['index']==rec['index'] and r['epoch']==budget)
            endpoints.append(meta|dict(budget=budget,epochs_run=min(budget,n),full_run_epochs=n,
                best_epoch=q['selected_epoch'],best_validation_loss=q['best_validation_loss'],
                test_accuracy=q['test_accuracy'],test_loss=q['test_cross_entropy'],ordinary_stopping_reached=True))
        final_metric=json.loads((folder/'performance/final.json').read_text())
        last=trajectories[-1]
        assert abs(last['test_accuracy']-final_metric['accuracy']['test'])<1e-12
        assert abs(last['test_cross_entropy']+final_metric['categorical_loglikelihood']['test'])<1e-12
        print('Read selected metrics',rec['index'],len(selected_cache),flush=True)
    d=pd.DataFrame(trajectories);end=pd.DataFrame(endpoints)
    pd.DataFrame(histories).to_csv(out/'observed_training_histories.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'checkpoint_audit.csv',index=False)
    pd.DataFrame(metric_audit).to_csv(out/'selected_metric_audit.csv',index=False)
    d.to_csv(out/'validation_selected_seed_trajectories.csv',index=False)
    end.to_csv(out/'endpoints.csv',index=False)
    stops=pd.DataFrame(checks)[['index','arm','depth','seed','epochs','best_epoch','cap','ordinary_stopping_reached']].rename(columns={'epochs':'epochs_run'})
    stops.to_csv(out/'stopping_by_seed.csv',index=False)
    ix=np.random.default_rng(601806).integers(10,size=(10000,10))
    weights=np.array([np.bincount(i,minlength=10) for i in ix],float)/10
    summaries=[];paired=[];paired_seed=[];contrasts=[]
    for (arm,depth),group in d.groupby(['arm','depth']):
        assert set(group.seed)==set(range(10200,10210))
        for metric in ['test_accuracy','test_cross_entropy','best_validation_loss']:
            v=group.pivot(index='seed',columns='epoch',values=metric).sort_index()
            assert not v.isna().any().any()
            mean,lo,hi=bounds(v.to_numpy(),weights)
            summaries.extend(dict(arm=arm,depth=int(depth),metric=metric,epoch=int(e),mean=float(m),ci95_low=float(l),ci95_high=float(h),n_seeds=10) for e,m,l,h in zip(v.columns,mean,lo,hi))
    local=d[d.depth.eq(3)&d.arm.isin(['path_transport','per_soma_shared'])]
    for metric in ['test_accuracy','test_cross_entropy','best_validation_loss']:
        p=local.pivot(index=['seed','epoch'],columns='arm',values=metric)
        v=(p.path_transport-p.per_soma_shared).unstack('epoch').sort_index()*(100 if metric=='test_accuracy' else 1)
        mean,lo,hi=bounds(v.to_numpy(),weights)
        paired.extend(dict(metric=metric,epoch=int(e),mean=float(m),ci95_low=float(l),ci95_high=float(h),n_seeds=10,positive_seeds=int((v[e]>0).sum()),negative_seeds=int((v[e]<0).sum())) for e,m,l,h in zip(v.columns,mean,lo,hi))
        paired_seed.extend(dict(metric=metric,seed=int(seed),epoch=int(e),exact_minus_shared=float(v.loc[seed,e])) for seed in v.index for e in v.columns)
    for budget,group in end.groupby('budget'):
        for metric in ['test_accuracy','test_loss']:
            p=group.pivot(index='seed',columns=['arm','depth'],values=metric).sort_index()
            for name,left,right in CONTRASTS:
                v=(p[left]-p[right]).to_numpy()*(100 if metric=='test_accuracy' else 1)
                m,l,h=bounds(v,weights)
                contrasts.append(dict(budget=int(budget),metric=metric,contrast=name,mean=float(m),ci95_low=float(l),ci95_high=float(h),n_seeds=10,positive_seeds=int((v>0).sum()),negative_seeds=int((v<0).sum()),units='percentage points' if metric=='test_accuracy' else 'nats'))
    # Report differences from the older mixed-device 600-epoch restarts.
    historical = Path(protocol['records'][0]['original_config']).parents[2] / 'physical_depth_followup/validation_selected_seed_trajectories.csv'
    if historical.is_file():
        old = pd.read_csv(historical, float_precision='round_trip')
        keys = ['arm', 'depth', 'seed', 'epoch']
        columns = ['test_accuracy', 'test_cross_entropy', 'best_validation_loss']
        comparison = d[d.epoch.isin([180, 600])][keys + columns].merge(
            old[old.epoch.isin([180, 600])][keys + columns], on=keys,
            suffixes=('_uniform_h200', '_historical'), validate='one_to_one')
        assert len(comparison) == 120
        for metric in columns:
            comparison[metric + '_difference'] = comparison[metric + '_uniform_h200'] - comparison[metric + '_historical']
        comparison.to_csv(out / 'historical_restart_comparison.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/'condition_trajectory_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_trajectory_summary.csv',index=False)
    pd.DataFrame(paired_seed).to_csv(out/'paired_seed_trajectories.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    dump(out/'summary.json',dict(integrity_valid=True,stopping_valid=True,n_expected=60,n_complete=60,
        max_observed_epoch=maxepoch,display_epoch_count=len(epochs),all_seeds_retained=True,
        protocol_sha256=sha(args.protocol),analyzer_sha256=sha(__file__),
        scope='Post-review same-seed uniform-H200 extension; original records retained; no fresh confirmation. Ordinary validation early stopping is not a proof of global convergence.',
        curve_sampling='Every epoch through 600, then 256 log-spaced epochs plus budget and stopping epochs; validation-best states retained after stopping.',
        bootstrap='10000 paired whole-seed draws, seed 601806; descriptive pointwise intervals',
        endpoint_contrasts=contrasts))

if __name__=='__main__':main()
