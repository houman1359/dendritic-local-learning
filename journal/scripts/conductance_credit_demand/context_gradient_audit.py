#!/usr/bin/env python3
"""Post-fit mechanistic diagnostics at matched parameter states, never training input."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import opponent_model as model
from run_opponent import OUT,sha,write

def cosine(a,b): return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
def audit():
 cfg=json.loads((OUT/'protocol.json').read_text());selection=json.loads((OUT/'selection_freeze.json').read_text());summary=[];components=[];worst=0.;matched=0
 for seed in cfg['fresh_seeds']:
  for name in selection['confirmatory_tasks']:
   folder=OUT/'runs/fresh';meta=json.loads((folder/f'seed_{seed}_{name}_metadata.json').read_text());task=meta['task'];x,y,context=model.data(seed,'diagnostic',cfg['n_diagnostic'],task['conflict'],task['gate']);variance=meta['training_target_variance'];primary=np.load(folder/f'seed_{seed}_{name}_states.npz');extension=np.load(OUT/'extension'/f'seed_{seed}_{name}_states.npz');states=[]
   for i,r in enumerate(meta['records']):
    if r['optimizer']!='adam' or not r['selected_rate']:continue
    for label,theta in [('initial',primary['theta'][0,i]),('step256',primary['theta'][2,i]),('primary_best',primary['best_theta'][i]),('extended_best',extension['best_theta'][i])]: states.append((r['rule'],label,theta,primary['initial_profiles'][i]))
   for source,label,theta,profile in states:
    t=theta[None];s=model.forward(t,x);e=model.eligibility(t,x)[0];q=s['path'][0];error=(s['output'][0]-y)/variance;exact_sample=error[:,None]*e*q[:,model.PARAM_UNIT];exact=exact_sample.mean(0)
    for rule in model.RULES:
     routed=q.copy()
     if rule=='unit_broadcast':routed[:,:6]=1.
     elif rule=='calibrated_broadcast':routed[:,:6]=profile
     elif rule=='ancestry_three_oracle':routed[:,:6]=model.project_ancestry(q[:,:6],profile)
     sample=error[:,None]*e*routed[:,model.PARAM_UNIT];g=sample.mean(0);expected,_=model.gradients(t,x,y,variance,profile[None],[rule]);worst=max(worst,float(np.max(abs(g-expected[0]))));matched+=1
     for scope,ix in [('all_parameters',np.arange(24)),('distal_parameters',np.arange(16))]:
      percontext=[sample[context==c][:,ix].mean(0) for c in [0,1]];weighted=[a*np.mean(context==c) for c,a in enumerate(percontext)];den=sum(np.linalg.norm(a) for a in weighted);cancellation=float(np.linalg.norm(sum(weighted))/max(den,1e-30));perdot=np.sum(sample[:,ix]*exact_sample[:,ix],axis=1);norms=np.linalg.norm(sample[:,ix],axis=1)*np.linalg.norm(exact_sample[:,ix],axis=1)
      summary.append(dict(seed=seed,task=name,source_rule=source,state=label,evaluated_rule=rule,parameter_scope=scope,gradient_cosine=cosine(g[ix],exact[ix]),exact_gradient_norm=float(np.linalg.norm(exact[ix])),delivered_gradient_norm=float(np.linalg.norm(g[ix])),context_mean_gradient_cosine=cosine(percontext[0],percontext[1]),context_cancellation_ratio=cancellation,mean_example_gradient_cosine=float(np.mean(perdot/np.maximum(norms,1e-30))),min_example_gradient_inner_product=float(perdot.min()),diagnostic_nmse=float(np.mean((s['output'][0]-y)**2)/np.var(y))))
     for c in [0,1]:
      cg=sample[context==c].mean(0)
      for j,value in enumerate(cg): components.append(dict(seed=seed,task=name,source_rule=source,state=label,evaluated_rule=rule,context=c,parameter=j,compartment=int(model.PARAM_UNIT[j]),gradient=float(value),n_examples=int(np.sum(context==c))))
 out=OUT/'summaries';pd.DataFrame(summary).to_csv(out/'context_gradient_summary.csv',index=False);pd.DataFrame(components).to_csv(out/'context_gradient_components.csv.gz',index=False,compression='gzip');write(out/'context_gradient_audit.json',dict(matched_gradient_checks=matched,maximum_absolute_error=worst,scope='Shared model parameters and inputs per evaluated rule; diagnostic set never used for training or selection.',source_sha256=sha(Path(__file__))))
 print('checks',matched,'max error',worst);df=pd.DataFrame(summary);print(df[(df.task==selection['selected_task'])&(df.parameter_scope=='distal_parameters')&(df.source_rule=='calibrated_broadcast')&(df.state=='extended_best')].groupby('evaluated_rule')[['gradient_cosine','exact_gradient_norm','delivered_gradient_norm','context_mean_gradient_cosine','context_cancellation_ratio']].mean().to_string())

if __name__=='__main__':audit()
