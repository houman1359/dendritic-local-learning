"""Nested linear baselines and explicit observed-input sensitivity, with no new nonlinear fits."""
from __future__ import annotations
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import run_reconstructed_tree_task_learning as base
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/review_completion/measured_response_baselines.json'
OUT=ROOT/'source_data/review_response_baselines'
KEY=['target_root_id','session','scan_idx','replicate']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reliability(x,stim):
    aa=[];bb=[]
    for c in np.unique(stim):
        ix=np.flatnonzero(stim==c)
        if len(ix)>1:aa.append(x[ix[::2]].mean(0));bb.append(x[ix[1::2]].mean(0))
    if len(aa)<3:return np.full(x.shape[1],np.nan)
    a,b=np.asarray(aa),np.asarray(bb)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray([spearmanr(a[:,j],b[:,j]).statistic for j in range(x.shape[1])])

def outer_and_inner(data,rep,cfg,old):
    root=data['root_id'];stream=100*data['session']+data['scan_idx']
    train,test=base.grouped_split(data['stimulus_ids'],base.stable_rng(old['seed'],root,1000+stream,rep),old['test_fraction'])
    identities=np.unique(data['stimulus_ids'][train]);rng=base.stable_rng(cfg['analysis_seed'],root,1000+stream,rep)
    partitions=np.array_split(rng.permutation(identities),cfg['inner_folds']);folds=[]
    for ids in partitions:
        val=train & np.isin(data['stimulus_ids'],ids);inner=train & ~val
        assert not(np.intersect1d(data['stimulus_ids'][inner],data['stimulus_ids'][val]).size)
        assert not(np.intersect1d(data['stimulus_ids'][inner|val],data['stimulus_ids'][test]).size)
        folds.append((inner,val))
    return train,test,folds

def design(data,train,condition,old,noise):
    x=data['x_raw'];p=x.shape[1];mask=np.ones(p,dtype=bool)
    if condition['mask']=='manual':mask=data['contacts']['manual_match'].fillna(False).astype(bool).to_numpy()
    elif condition['mask']=='reliable':mask=np.nan_to_num(reliability(x[train],data['stimulus_ids'][train]),nan=-2)>=condition['threshold']
    elif condition['mask']=='drop':
        mask=np.zeros(p,dtype=bool);mask[condition['order'][:max(1,int(round(p*condition['retain'])))]]=True
    if condition.get('raw',False):z=x[:,mask].copy()
    elif not mask.any():z=np.empty((len(x),0))
    else:z=base.scale_inputs(x[:,mask],train,old['input_lower_quantile'],old['input_upper_quantile'],old['input_maximum'])
    if z.shape[1]:
        mu=z[train].mean(0);sd=z[train].std(0,ddof=1);sd=np.where(sd>1e-8,sd,1.)
        z=(z-mu)/sd
        z+=condition.get('noise',0)*noise[:,mask]
    return z,mask

def fit_predict(x,y,train,test,penalties):
    # Recenter after supplied perturbations; an unpenalized intercept is included.
    xx=x[train];yy=y[train];mu=xx.mean(0);my=yy.mean();xc=xx-mu;yc=yy-my
    if not x.shape[1]:return np.full((len(penalties),int(test.sum())),my)
    gram=xc.T@xc/len(xc);rhs=xc.T@yc/len(xc);ev,v=np.linalg.eigh(gram);ev=np.maximum(ev,0)
    coeff=[]
    for alpha in penalties:
        denom=ev+alpha;inverse=np.zeros_like(denom);good=denom>max(ev.max(),1.)*1e-12
        inverse[good]=1/denom[good];coeff.append(v@(inverse*(v.T@rhs)))
    return np.asarray(coeff)@(x[test]-mu).T+my

def one(data,rep,cfg,old,recorded):
    root=data['root_id'];stream=100*data['session']+data['scan_idx'];train,test,folds=outer_and_inner(data,rep,cfg,old)
    hist=recorded[recorded.target_root_id.eq(root)&recorded.session.eq(data['session'])&recorded.scan_idx.eq(data['scan_idx'])&recorded.replicate.eq(rep)]
    assert len(hist)==4
    assert hist.n_train_trials.eq(train.sum()).all() and hist.n_test_trials.eq(test.sum()).all()
    rng=base.stable_rng(cfg['analysis_seed'],root,2000+stream,rep);noise=rng.normal(size=data['x_raw'].shape)
    conditions=[dict(method='training_mean',mask='all'),dict(method='ols_all',mask='all'),dict(method='ridge_all',mask='all'),dict(method='ridge_raw',mask='all',raw=True),dict(method='manual_only',mask='manual')]
    conditions += [dict(method=f'training_reliability_ge_{v:g}',mask='reliable',threshold=v) for v in [0.,.1,.2]]
    for d in range(cfg['mask_draws']):
        order=rng.permutation(data['x_raw'].shape[1])
        conditions += [dict(method=f'keep_{frac:g}',mask='drop',retain=frac,draw=d,order=order) for frac in cfg['retained_input_fractions']]
    conditions += [dict(method=f'noise_{n:g}',mask='all',noise=n) for n in cfg['added_feature_noise_sd']]
    y=base.scale_target(data['y_raw'],train);denom=np.mean((y[test]-y[train].mean())**2)
    rows=[]
    for condition in conditions:
        method=condition['method'];penalties=[0.] if method=='ols_all' else cfg['ridge_penalties'];scores=[]
        if method=='training_mean':pred=np.full(test.sum(),y[train].mean());alpha=np.nan;nfeatures=0
        else:
            for inner,val in folds:
                x,_=design(data,inner,condition,old,noise);prediction=fit_predict(x,y,inner,val,penalties)
                scores.append(np.mean((prediction-y[val])**2,axis=1))
            losses=np.mean(scores,axis=0);best=int(np.argmin(losses));alpha=penalties[best]
            x,mask=design(data,train,condition,old,noise);nfeatures=int(mask.sum());pred=fit_predict(x,y,train,test,[alpha])[0]
        rows.append(dict(target_root_id=root,session=data['session'],scan_idx=data['scan_idx'],replicate=rep,method=method,draw=condition.get('draw',0),n_features=nfeatures,n_available=data['x_raw'].shape[1],selected_penalty=alpha,nmse=np.mean((pred-y[test])**2)/denom,n_train_trials=int(train.sum()),n_test_trials=int(test.sum())))
    allreliability=reliability(np.column_stack([data['x_raw'],data['y_raw']])[train],data['stimulus_ids'][train])
    rel=dict(target_root_id=root,session=data['session'],scan_idx=data['scan_idx'],replicate=rep,median_partner_reliability=np.nanmedian(allreliability[:-1]),target_reliability=allreliability[-1],n_manual=int(data['contacts']['manual_match'].sum()),n_partners=data['x_raw'].shape[1])
    return rows,rel

def interval(values,rng,draws):
    values=np.asarray(values);m=values[rng.integers(0,len(values),(draws,len(values)))].mean(1);return [float(values.mean()),*np.quantile(m,[.025,.975]).tolist()]

def main():
    cfg=json.loads(CONFIG.read_text());old=json.loads((ROOT/'source_data/fulltree_boundary/output/config.json').read_text());recorded=pd.read_csv(ROOT/'source_data/fulltree_boundary/output/runs.csv');OUT.mkdir(exist_ok=True,parents=True)
    start=time.monotonic();rows=[];rels=[];manifest=[]
    for path in sorted(base.DEFAULT_EXTRACT_ROOT.glob('target*_automatic_conservative')):
        data=base.load_target(path);assert 'manual_match' in data['contacts']
        for rep in range(old['replicates']):
            a,b=one(data,rep,cfg,old,recorded);rows.extend(a);rels.append(b)
        manifest.append(dict(root=data['root_id'],session=data['session'],scan_idx=data['scan_idx'],files=data['input_files']))
        print(path.name,len(rows),round(time.monotonic()-start,1),flush=True)
    frame=pd.DataFrame(rows);assert frame.nmse.notna().all() and np.isfinite(frame.nmse).all()
    run=frame.groupby(KEY+['method'],as_index=False)[['nmse','n_features','n_available']].mean()
    for method in recorded.method.unique():
        h=recorded[recorded.method.eq(method)][KEY+['heldout_normalized_mse','n_sites']].rename(columns={'heldout_normalized_mse':'nmse','n_sites':'n_features'})
        h['method']='archived_'+method;h['n_available']=h.n_features;run=pd.concat([run,h],ignore_index=True)
    assert run.groupby(KEY+['method']).size().eq(1).all()
    scan=run.groupby(KEY[:-1]+['method'],as_index=False)[['nmse','n_features','n_available']].mean();cell=scan.groupby(['target_root_id','method'],as_index=False)[['nmse','n_features','n_available']].mean()
    rng=np.random.default_rng(cfg['analysis_seed']);summary=[]
    for method,g in cell.groupby('method'):
        mean,lo,hi=interval(g.nmse,rng,cfg['bootstrap_draws']);summary.append(dict(method=method,n_targets=len(g),mean_nmse=mean,ci95_low=lo,ci95_high=hi,mean_features=g.n_features.mean()))
    summary=pd.DataFrame(summary);contrasts=[];wide=cell.pivot(index='target_root_id',columns='method',values='nmse')
    for method in wide.columns:
        if method=='ridge_all':continue
        d=wide[method]-wide.ridge_all;mean,lo,hi=interval(d,rng,cfg['bootstrap_draws']);contrasts.append(dict(comparator=method,n_targets=len(d),other_minus_allridge_mean_nmse=mean,ci95_low=lo,ci95_high=hi,positive_targets=int((d>0).sum())))
    for name,t in [('perturbation_runs',frame),('run_metrics',run),('scan_metrics',scan),('target_metrics',cell),('condition_summary',summary),('paired_ridge_contrasts',pd.DataFrame(contrasts)),('training_only_reliability',pd.DataFrame(rels))]:t.to_csv(OUT/f'{name}.csv',index=False,float_format='%.12g')
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(complete=True,outer_splits=len(run[KEY].drop_duplicates()),n_scans=len(scan[KEY[:-1]].drop_duplicates()),n_targets=cell.target_root_id.nunique(),nested_tuning=True,all_outer_stimulus_splits_disjoint=True,n_new_evaluations=len(frame),runtime_seconds=time.monotonic()-start,protocol_sha256=sha(CONFIG),script_sha256=sha(__file__),historical_runs_sha256=sha(ROOT/'source_data/fulltree_boundary/output/runs.csv'),conditions=summary.to_dict('records'),ridge_contrasts=contrasts,scope=cfg['scope'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'README.md').write_text('# Measured-response baselines and observed-input sensitivity\n\n'+cfg['scope']+'\n\nRun `OPENBLAS_NUM_THREADS=1 python scripts/analyze_review_response_baselines.py`. All preprocessing, repeat-reliability masks and ridge tuning use training identities only. Manual masks use archived contact provenance. Missingness masks are nested within each of five random orderings; perturbation draws average before outer splits, scans and seven targets. Noise is added to predictors in train and test, scaled by training-only feature SD; this is measurement degradation, not biological time or a task-rank manipulation. Empty masks use an intercept-only model and remain included. Archived nonlinear fits are read unchanged. Output rows retain every declared condition. Intervals resample seven target means; no population-power or equivalence claim is made.\n')
    print(summary.to_string(index=False));print(json.dumps(contrasts[:5],indent=2))
if __name__=='__main__':main()
