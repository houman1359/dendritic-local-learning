#!/usr/bin/env python3
"""Summarize the complete frozen sensitivity study without fitting or selecting a model."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from model import exact_test_lattice,load_dataset
J=Path(__file__).resolve().parents[2];OUT=J/'source_data/measured_alignment_power'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h

def threshold(table,target=.8):
 table=table.sort_values('lambda');y=table.power.to_numpy();x=table['lambda'].to_numpy();e=table.mean_target_effect.to_numpy()
 inds=[k for k in range(len(y)) if np.all(y[k:]>=target)]
 if not inds:return dict(reachable=False,maximal_specified_signal_power=float(y[-1]),maximal_signal_power_ci95=[float(table.iloc[-1].power_ci95_low),float(table.iloc[-1].power_ci95_high)],statement='80% detection is not sustained/reached in the tested signal family; no detectable-correlation threshold is claimed.')
 k=inds[0];lower=max(k-1,0);t=0 if k==0 else (target-y[lower])/(y[k]-y[lower]);estimated=x[lower]+t*(x[k]-x[lower]);effect=e[lower]+t*(e[k]-e[lower])
 lowers=table.power_ci95_low.to_numpy();strong=[m for m in range(len(y)) if np.all(lowers[m:]>=target)]
 return dict(reachable=True,smallest_sustained_tested_lambda=float(x[k]),lambda_grid_bracket=[float(x[lower]),float(x[k])],descriptive_interpolated_lambda=float(estimated),descriptive_mean_partial_rank_at_interpolation=float(effect),smallest_lambda_lower_mc_bound_ge_80=float(x[strong[0]]) if strong else None,scope='Interpolation of this fixed simulated Gaussian ancestry-variance family and observed sampling, not a universal detectable biological correlation.')

def main():
 protocol=json.loads((OUT/'protocol_freeze.json').read_text());ph=sha(OUT/'protocol_freeze.json');chunks=[];inputfiles=[OUT/'protocol_freeze.json',OUT/'PROTOCOL.md',OUT/'input_manifest.json',OUT/'input_audit.json',Path(__file__),J/'scripts/measured_alignment_power/model.py',J/'scripts/figure_canvas.py',J/'scripts/journal_style.py'];execution=[]
 for k in range(protocol['n_chunks']):
  file=OUT/'runs'/f'chunk_{k:02d}.npz';meta=file.with_suffix('.json');r=json.loads(meta.read_text());assert r['status']=='complete' and r['protocol_sha256']==ph and r['output_sha256']==sha(file)
  c=np.load(file,allow_pickle=False);assert np.array_equal(c['replicate_ids'],np.arange(k*protocol['replicates_per_chunk'],(k+1)*protocol['replicates_per_chunk']));chunks.append(c);execution.append(r);inputfiles+=[file,meta]
 scan=np.concatenate([c['scan_effects'] for c in chunks],axis=2);targets=np.concatenate([c['target_effects'] for c in chunks],axis=2);ps=np.concatenate([c['exact_two_sided_p'] for c in chunks],axis=2);assert np.isfinite(scan).all() and np.isfinite(targets).all()
 mean=targets.mean(-1);positive=(ps<=.05)&(mean>0);n=targets.shape[2];rows=[];targetrows=[]
 for i,lam in enumerate(protocol['ancestry_variance_fractions']):
  for j,name in enumerate(protocol['reliability_scenarios']):
   k=int(positive[i,j].sum());lo,hi=wilson(k,n);twosided=int((ps[i,j]<=.05).sum());two_lo,two_hi=wilson(twosided,n)
   rows.append(dict(reliability=name,lambda_value=lam,**{'lambda':lam},n_replicates=n,n_positive_detections=k,power=k/n,power_ci95_low=lo,power_ci95_high=hi,mean_target_effect=float(mean[i,j].mean()),target_effect_sd_across_datasets=float(mean[i,j].std(ddof=1)),mean_effect_mc_se=float(mean[i,j].std(ddof=1)/np.sqrt(n)),n_two_sided_rejections=twosided,two_sided_rejection_rate=twosided/n,two_sided_ci95_low=two_lo,two_sided_ci95_high=two_hi))
   if lam in [0,.5,1]:
    for t,root in enumerate(chunks[0]['target_root_ids']):targetrows.append(dict(reliability=name,**{'lambda':lam},target_root_id=int(root),mean_partial_rank=float(targets[i,j,:,t].mean()),sd=float(targets[i,j,:,t].std(ddof=1)),positive_fraction=float((targets[i,j,:,t]>0).mean()),p05=float(np.quantile(targets[i,j,:,t],.05)),p95=float(np.quantile(targets[i,j,:,t],.95))))
 table=pd.DataFrame(rows);table.to_csv(OUT/'power_summary.csv',index=False);pd.DataFrame(targetrows).to_csv(OUT/'target_sensitivity_summary.csv',index=False)
 null={};dep={};thresholds={}
 for name in protocol['reliability_scenarios']:
  j=protocol['reliability_scenarios'].index(name);r=table[table.reliability.eq(name)&table['lambda'].eq(0)].iloc[0];null[name]=dict(two_sided_typeI=float(r.two_sided_rejection_rate),ci95=[float(r.two_sided_ci95_low),float(r.two_sided_ci95_high)],positive_direction_false_detection=float(r.power),positive_ci95=[float(r.power_ci95_low),float(r.power_ci95_high)],warning_lower_interval_exceeds_nominal=bool(r.two_sided_ci95_low>.05))
  corr=np.corrcoef(targets[0,j].T);scorr=np.corrcoef(scan[0,j].T);np.savetxt(OUT/f'null_target_correlation_{name}.csv',corr,delimiter=',');np.savetxt(OUT/f'null_scan_correlation_{name}.csv',scorr,delimiter=',');dep[name]=dict(max_abs_offdiagonal_target_correlation=float(np.max(np.abs(corr-np.eye(7)))),scope='Finite Monte Carlo correlation among target null statistics; does not replace the joint simulation or prove biological independence.')
  thresholds[name]=threshold(table[table.reliability.eq(name)])
 calpath=OUT/'reliability_calibration_audit.csv';cal=pd.read_csv(calpath);inputfiles+=[calpath,OUT/'reliability_calibration_audit.json'];difference=cal.simulated_mean_split_half_spearman-cal.clipped_measured_spearman
 result=dict(status='complete',protocol_sha256=ph,n_global_replicates=n,n_scans=13,n_targets=7,n_unique_partners=102,n_partner_observations=125,n_signal_levels=len(protocol['ancestry_variance_fractions']),reliability_scenarios=protocol['reliability_scenarios'],null_calibration=null,thresholds=thresholds,exact_seven_target_test=exact_test_lattice(),dependency_audit=dep,reliability_calibration=dict(n_partner_records=len(cal),mean_absolute_spearman_discrepancy=float(np.abs(difference).mean()),max_absolute_spearman_discrepancy=float(np.abs(difference).max()),mean_signed_discrepancy=float(difference.mean()),scope='Gaussian pooled-moment approximation checked without retuning; this is simulation calibration, not new observed reliability.'),execution=dict(n_completed_chunks=len(execution),sum_cpu_task_seconds=float(sum(v['seconds'] for v in execution)),slurm_job_ids=sorted(set(v['slurm_job_id'] for v in execution))),assumptions='Conditional Gaussian ancestry-kernel tuning on observed partners, per-recording measured-repeat attenuation and finite stimulus set. No inference about unobserved inputs, arbitrary true effect shapes or endogenous teaching.')
 (OUT/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Conditional sensitivity results','',f'Protocol `{ph}`; {n:,} simulated global datasets, 13 scans nested in seven targets. All original observed outcomes are unchanged.','']
 for name in protocol['reliability_scenarios']:
  d=thresholds[name];rr=null[name];last=table[table.reliability.eq(name)&table['lambda'].eq(1)].iloc[0]
  lines += [f'## {name.capitalize()} reliability','',f'At maximal specified ancestry variance (lambda=1), positive-alignment detection is {last.power:.3f} (95% Monte Carlo interval {last.power_ci95_low:.3f}–{last.power_ci95_high:.3f}). The corresponding mean seven-target partial-rank effect is {last.mean_target_effect:.3f}.']
  if d['reachable']:lines += [f"The smallest sustained tested 80% crossing is lambda={d['smallest_sustained_tested_lambda']:.2f}, bracketed by {d['lambda_grid_bracket']}. Descriptive interpolation gives lambda={d['descriptive_interpolated_lambda']:.3f} and mean observed partial-rank effect {d['descriptive_mean_partial_rank_at_interpolation']:.3f}. This is a model-conditional sensitivity number, not a universal detectable correlation."]
  else:lines += [d['statement']]
  lines += [f"Zero-signal two-sided type I error is {rr['two_sided_typeI']:.3f} (Wilson95% {rr['ci95'][0]:.3f}–{rr['ci95'][1]:.3f}); positive-direction false detection is {rr['positive_direction_false_detection']:.3f}.",'']
 lines+=['## Interpretation and limits','', 'The exact seven-unit test has minimum two-sided P=0.015625 and rejects six of 128 untied sign patterns at 0.05. This discreteness is not a universal ceiling on power. Power at lambda=1 is maximal within the specified variance-mixture grid, not across all possible biological signals. All response pairs are computed from valid joint response matrices, and known shared roots/recordings are preserved. The calculation remains conditional on the observed sampling and Gaussian/reliability model; it does not establish absence of biological ancestry alignment or sensitivity to missing branches.','',f"The reliability calibration audit has mean absolute Spearman discrepancy {result['reliability_calibration']['mean_absolute_spearman_discrepancy']:.4f} and maximum {result['reliability_calibration']['max_absolute_spearman_discrepancy']:.4f}; no calibration parameters were adjusted after inspecting these outcomes."]
 if any(v['warning_lower_interval_exceeds_nominal'] for v in null.values()):lines+=['','WARNING: A zero-signal type I interval lies above nominal 0.05. Treat the associated sensitivity curve as nominal and potentially anticonservative; no threshold correction was fitted after outcomes.']
 (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 fig=draw(table,cal)
 provenance=dict(protocol_sha256=ph,source_sha256={str(p.relative_to(J)):sha(p) for p in inputfiles},summary_files={str(p.relative_to(J)):sha(p) for p in [OUT/'power_summary.csv',OUT/'target_sensitivity_summary.csv',OUT/'RESULTS.json']},figure_files={str(p.relative_to(J)):sha(p) for p in fig})
 (OUT/'figure_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(result,indent=2))

def draw(table,cal):
 sys.path.insert(0,str(J/'scripts'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from figure_canvas import NativeCanvas,Margins,COLORS,style_panel,PT_SMALL,LW_REF,LW_ERR
 canvas=NativeCanvas(410/72,2,row_weights=[140,125],hgutter_pt=30,vgutter_pt=53,margins=Margins(left=51,right=20,top=30,bottom=42))
 a=canvas.panel('A',0,0,6,title='Detection with the observed sampling');b=canvas.panel('B',0,6,6,title='Detected partial-rank effect');c=canvas.panel('C',1,0,12,title='Measured reliability sets response noise')
 for name,color,marker in [('measured',COLORS['shunting'],'o'),('perfect',COLORS['mute'],'s')]:
  f=table[table.reliability.eq(name)].sort_values('lambda');x=f['lambda'].to_numpy();y=f.power.to_numpy();label='Measured reliability' if name=='measured' else 'Perfect reliability'
  a.plot(x,y,color=color,marker=marker,ms=2.5,lw=1,label=label);a.fill_between(x,f.power_ci95_low.to_numpy(),f.power_ci95_high.to_numpy(),color=color,alpha=.14,lw=0)
  b.plot(f.mean_target_effect,y,color=color,marker=marker,ms=2.5,lw=1)
  b.fill_between(f.mean_target_effect.to_numpy(),f.power_ci95_low.to_numpy(),f.power_ci95_high.to_numpy(),color=color,alpha=.14,lw=0)
 for ax in [a,b]:ax.axhline(.8,color=COLORS['mute'],ls='--',lw=LW_REF);ax.set_ylim(0,1.03);ax.set_yticks([0,.2,.4,.6,.8,1]);ax.set_ylabel('Positive-alignment detection probability');style_panel(ax,grid='y')
 a.set(xlabel='Ancestry fraction of reliable tuning variance',xlim=(0,1),xticks=[0,.25,.5,.75,1]);a.legend(frameon=False,fontsize=PT_SMALL,loc='lower right')
 b.set_xlabel('Simulated mean seven-target partial rank r')
 cc=cal.measured_split_half_spearman.to_numpy();yy=cal.simulated_mean_split_half_spearman.to_numpy();c.scatter(cc,yy,s=12,color=COLORS['shunting'],alpha=.6,edgecolors='none');neg=cc<0;c.scatter(cc[neg],yy[neg],s=26,color=COLORS['scalar'],marker='s')
 lo=min(-.03,cc.min()-.025);hi=max(cc.max(),yy.max())+.025;c.plot([0,hi],[0,hi],color=COLORS['mute'],ls='--',lw=LW_REF);c.plot([lo,0],[0,0],color=COLORS['mute'],ls=':',lw=LW_REF)
 c.set(xlabel='Measured odd/even repeat reliability (Spearman r)',ylabel='Model mean repeat reliability',xlim=(lo,hi),ylim=(-.025,hi));style_panel(c,grid='y');c.text(.02,.94,'125 partner records; negative estimates set to zero reliable variance',transform=c.transAxes,fontsize=PT_SMALL,ha='left',va='top')
 dest=OUT/'figures';dest.mkdir(exist_ok=True);pdf=dest/'measured_alignment_sensitivity.pdf';canvas.save(pdf,name='measured_alignment_sensitivity',dpi=180)
 return [pdf,pdf.with_suffix('.png')]
if __name__=='__main__':main()
