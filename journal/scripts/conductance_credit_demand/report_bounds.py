#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_opponent import OUT,sha,write
from report import interval

def report():
 root=OUT/'bound_sensitivity';cfg=json.loads((OUT/'protocol.json').read_text());dfs={k:[] for k in ['endpoints','curves','contacts']};errors=[]
 for seed in cfg['fresh_seeds']:
  folder=root/'runs';audit=json.loads((folder/f'seed_{seed}_audit.json').read_text());errors.append(audit['maximum_original_replay_parameter_error'])
  for f,h in audit['files_sha256'].items():assert sha(folder/f)==h
  for k in dfs:dfs[k].append(pd.read_csv(folder/f'seed_{seed}_{k}.csv'))
 out=root/'summaries';out.mkdir(exist_ok=True);data={k:pd.concat(v,ignore_index=True) for k,v in dfs.items()}
 for k,d in data.items():d.to_csv(out/f'all_{k}.csv',index=False)
 ends=data['endpoints'];contrasts=[];paired=[]
 for bound in [7.,20.]:
  for opt in cfg['rates']:
   for rule in [r for r in cfg['rules'] if r!='exact']:
    part=ends[(ends.bound==bound)&(ends.optimizer==opt)&ends.selected_rate];w=part.pivot(index='seed',columns=['task','rule'],values='test_nmse');tg=w[('opposed_strong',rule)]-w[('opposed_strong','exact')];cg=w[('aligned_strong',rule)]-w[('aligned_strong','exact')]
    for name,values in [('target_gap',tg),('control_gap',cg),('task_difference_in_gap',tg-cg)]:
     mean,lo,hi=interval(values);contrasts.append(dict(bound=bound,optimizer=opt,rule=rule,contrast=name,mean=mean,ci_low=lo,ci_high=hi,n=len(values),n_positive=int((values>0).sum())))
     for seed,value in values.items():paired.append(dict(seed=seed,bound=bound,optimizer=opt,rule=rule,contrast=name,value=value))
 pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_seed_contrasts.csv',index=False)
 first=[];last=[];contact=data['contacts'];curves=data['curves'];
 for seed in cfg['fresh_seeds']:
  for opt in cfg['rates']:
   for rule in ['unit_broadcast','calibrated_broadcast']:
    part=contact[(contact.seed==seed)&(contact.bound==7)&(contact.task=='opposed_strong')&(contact.optimizer==opt)&(contact.rule==rule)&contact.selected_rate];hits=part[part.first_contact_step>0];step=int(hits.first_contact_step.min()) if len(hits) else 16385;indices=hits[hits.first_contact_step==step].parameter.tolist();first.append(dict(seed=seed,optimizer=opt,rule=rule,first_contact_step=step,parameters=','.join(str(i) for i in indices),compartments=','.join(str(int(i)) for i in hits[hits.first_contact_step==step].compartment.tolist())))
    cp=curves[(curves.seed==seed)&(curves.bound==7)&(curves.task=='opposed_strong')&(curves.optimizer==opt)&curves.selected_rate];laststep=int(cp[cp.step<step].step.max());a=cp[(cp.step==laststep)&(cp.rule==rule)].iloc[0];b=cp[(cp.step==laststep)&(cp.rule=='exact')].iloc[0];last.append(dict(seed=seed,optimizer=opt,rule=rule,first_contact_step=step,last_precontact_checkpoint=laststep,exact_nmse=float(b.test_nmse),broadcast_nmse=float(a.test_nmse),gap=float(a.test_nmse-b.test_nmse)))
 pd.DataFrame(first).to_csv(out/'first_contact.csv',index=False);pd.DataFrame(last).to_csv(out/'precontact_gaps.csv',index=False)
 summary=ends[ends.selected_rate].groupby(['bound','task','optimizer','rule'],as_index=False).test_nmse.mean();summary.to_csv(out/'selected_rate_means.csv',index=False)
 write(out/'audit.json',dict(n_sensitivity_fits=len(ends),n_original_bound_replays=int((ends.bound==7).sum()),n_wider_bound_fits=int((ends.bound==20).sum()),maximum_replay_parameter_error=max(errors),all_files_hash_verified=True,source_sha256=sha(Path(__file__))))
 print(summary.to_string(index=False));print(pd.DataFrame(contrasts).query("rule=='calibrated_broadcast' and contrast=='task_difference_in_gap'").to_string(index=False));print('max replay error',max(errors));print(pd.DataFrame(last).groupby(['optimizer','rule']).agg(first_min=('first_contact_step','min'),first_max=('first_contact_step','max'),pre_gap_mean=('gap','mean'),n_positive=('gap',lambda x:int((x>0).sum()))).to_string())

if __name__=='__main__':report()
