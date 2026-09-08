#!/usr/bin/env python3
"""Draw observed ancestry-response evidence and actual transfer geometry.

Frozen-data reanalysis only. The representative route matrix is selected by
median site count, with target/session/scan identifiers as deterministic ties.
All 13 scan supports and all seven target outcomes are retained as Source Data.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
J=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(J/'scripts'))
sys.path.insert(0,str(J/'code/reconstructed_tree'))
from figure_canvas import NativeCanvas, Margins, COLORS, PT_SMALL, PT_LABEL, PT_TICK, LW_ERR, LW_REF, MARKER_MS, style_panel
from analyze_microns_morphology_credit import ancestry_matrix, parent_map
from journal_style import style_direct_color_labels
S=J/'source_data'; OUT=J/'figures/components/credit_first_figure_08.pdf'; REC=S/'credit_first_figures'
ROUTE=COLORS['shunting']; GRAY=COLORS['mute']; EXACT=COLORS['bp']

def summary(v,seed):
 v=np.asarray(v,float); rng=np.random.default_rng(seed)
 lo,hi=np.quantile(rng.choice(v,size=(20000,len(v)),replace=True).mean(1),[.025,.975])
 return v.mean(),lo,hi

def forest(ax, rows, xlabel, xlim=None):
 for k,(label,values,estimate,color) in enumerate(rows):
  y=len(rows)-1-k; m,lo,hi=estimate
  ax.scatter(values,y+np.linspace(-.12,.12,len(values)),s=9,color=color,alpha=.40,edgecolors='none')
  ax.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt='D',ms=MARKER_MS,color=color,mfc='white',lw=LW_ERR,capsize=2)
 ax.set_yticks(range(len(rows)),[x[0] for x in rows][::-1]);ax.set_ylim(-.55,len(rows)-.45)
 ax.set_xlabel(xlabel);style_panel(ax,grid='x');ax.tick_params(axis='y',length=0,labelsize=PT_SMALL)
 ax.spines['left'].set_visible(False)
 if xlim:ax.set_xlim(*xlim)

def main():
 REC.mkdir(exist_ok=True)
 seg=pd.read_csv(S/'figure3/segment_metrics.csv')
 metadata=[json.loads(x) for x in (S/'fulltree_boundary/output/dictionary_and_validation_metadata.jsonl').read_text().splitlines()]
 support=[]; matrices={}
 for r in metadata:
  if r['replicate']!=0:continue
  _,parents,_=parent_map(seg[seg.root_id.eq(r['target_root_id'])])
  a=ancestry_matrix(r['site_segment_ids'],r['selected_route_segments'],parents)
  assert np.count_nonzero(a)==r['dictionary_nonzeros']['topology-matched routes']
  key=(r['target_root_id'],r['session'],r['scan_idx']);matrices[key]=(a,r)
  support.append(dict(target_root_id=key[0],session=key[1],scan_idx=key[2],n_sites=len(a),n_routes=a.shape[1],coverage=a.any(1).mean(),sites_per_route=a.sum()/a.shape[1],one_site_routes=bool((a.sum(0)==1).all())))
 support=pd.DataFrame(support).sort_values(['n_sites','target_root_id','session','scan_idx']).reset_index(drop=True)
 assert len(support)==13 and support.target_root_id.nunique()==7
 rep=support.iloc[len(support)//2];key=tuple(int(rep[k]) for k in ['target_root_id','session','scan_idx']);a,meta=matrices[key]
 # Ancestor supports contain actual zeros; no decorative nested bands.
 canvas=NativeCanvas(472/72,3,row_weights=[72,124,137],hgutter_pt=30,vgutter_pt=48,margins=Margins(left=62,right=14,top=25,bottom=42))
 axa=canvas.panel('A',0,0,12,title='Observed ancestry–response similarity')
 axb=canvas.panel('B',1,0,5,title='Realized four-route support',schematic=True)
 axc=canvas.panel('C',1,5,7,title='Coverage across the 13 scans')
 axd=canvas.panel('D',2,0,6,title='Response prediction')
 axe=canvas.panel('E',2,6,6,title='Fixed-profile fidelity')
 estimates=[]
 sums=pd.read_csv(S/'review_evidence_reanalysis/functional_native_contrasts.csv')
 vals=pd.read_csv(S/'review_evidence_reanalysis/functional_native_target_effects.csv')
 rows=[]
 for mode,label in [('selected_scans','Selected scans'),('scan_complete','All eligible scans')]:
  r=sums[sums.endpoint.eq('structure_function_partial_r')&sums.comparison.eq(mode)].iloc[0]
  v=vals[vals.endpoint.eq('structure_function_partial_r')&vals.comparison.eq(mode)].effect.to_numpy()
  rows.append((label,v,(r['mean'],r.ci95_low,r.ci95_high),ROUTE))
 forest(axa,rows,'Partial rank correlation',(-.65,.55));axa.axvline(0,color=GRAY,lw=LW_REF,ls='--')
 # Matrix uses an inset to preserve the schematic slot's aspect ratio.
 axb.set_axis_off(); inset=axb.inset_axes([.22,.1,.52,.85])
 inset.imshow(a,aspect='auto',interpolation='nearest',cmap=ListedColormap(['#f1f1f1',ROUTE]),vmin=0,vmax=1)
 inset.set_xticks(range(4),['1','2','3','4']);inset.set_yticks(range(len(a)),np.arange(1,len(a)+1));inset.tick_params(length=0,labelsize=PT_SMALL)
 inset.set_xlabel('Route',fontsize=PT_LABEL);inset.set_ylabel('Mapped input',fontsize=PT_LABEL)
 axc.scatter(np.arange(1,14),support.coverage*100,color=ROUTE,s=18)
 axc.set(ylim=(0,105),xticks=[1,4,7,10,13],xlabel='Scan (ordered by input count)',ylabel='Inputs reached (%)')
 axc.axhline(100,color=GRAY,lw=LW_REF,ls='--');style_panel(axc,grid='y')
 target=pd.read_csv(S/'review_response_baselines/target_metrics.csv')
 rows=[]
 for k,(name,label,color) in enumerate([('archived_exact compartment error','Exact',EXACT),('ridge_all','Ridge',GRAY),('archived_topology-matched routes','Ancestry',ROUTE),('archived_random anatomical routes','Random',GRAY),('archived_site-shuffled routes','Shuffled',GRAY)]):
  v=target[target.method.eq(name)].sort_values('target_root_id').nmse.to_numpy(); assert len(v)==7
  est=summary(v,260906+k);rows.append((label,v,est,color));estimates.append(dict(panel='D',method=name,mean=est[0],ci95_low=est[1],ci95_high=est[2]))
 forest(axd,rows,'Normalized MSE',(.58,1.02));axd.set_xticks([.6,.8,1])
 ridge_mean=target[target.method.eq('ridge_all')].nmse.mean()
 axd.axvline(ridge_mean,color=GRAY,lw=LW_REF,ls=':',zorder=0)
 oracle=pd.read_csv(S/'fulltree_within_span_oracle/cell_metrics.csv');osum=pd.read_csv(S/'fulltree_within_span_oracle/condition_summary.csv')
 rows=[]
 for name,mode,label,color in [('unprojected baseline transfer','frozen_baseline','Unrestricted\nfixed profile',GRAY),('topology-matched routes','frozen_baseline','Ancestry\nfixed profile',ROUTE),('topology-matched routes','trialwise_update_oracle','Ancestry\noracle amplitudes',ROUTE)]:
  r=osum[osum.method.eq(name)&osum['mode'].eq(mode)].iloc[0]
  v=oracle[oracle.method.eq(name)&oracle['mode'].eq(mode)].sort_values('target_root_id').update_match.to_numpy();assert len(v)==7
  rows.append((label,v,(r.mean_update_match,r.ci95_low,r.ci95_high),color))
 forest(axe,rows,'Update reconstruction',(-.05,1.08));axe.set_xticks([0,.5,1])
 style_direct_color_labels(canvas.fig);canvas.lock_reserves()
 for group in [('A','B','D'),('C','E')]:
  arts=[q['art'] for q in canvas._letters if q['letter'] in group];x=min(q.get_position()[0] for q in arts)
  for q in arts:q.set_position((x,q.get_position()[1]))
 problems=canvas.save(OUT,name='credit_first_figure_08',dpi=180,lock=False)
 support.to_csv(REC/'figure_08_support.csv',index=False);pd.DataFrame(estimates).to_csv(REC/'figure_08_prediction_summary.csv',index=False)
 np.savez_compressed(REC/'figure_08_actual_support.npz',matrix=a,site_ids=np.array(meta['site_segment_ids']),route_ids=np.array(meta['selected_route_segments']))
 files=[Path(__file__),S/'figure3/segment_metrics.csv',S/'fulltree_boundary/output/dictionary_and_validation_metadata.jsonl',S/'review_evidence_reanalysis/functional_native_contrasts.csv',S/'review_evidence_reanalysis/functional_native_target_effects.csv',S/'review_response_baselines/target_metrics.csv',S/'fulltree_within_span_oracle/cell_metrics.csv',S/'fulltree_within_span_oracle/condition_summary.csv']
 payload=dict(representative=dict(zip(['target_root_id','session','scan_idx'],key)),selection='Median mapped-input count, identifier ties; no outcome selection',coordinate_definition='Rows are mapped partner inputs; multiple inputs can share a physical segment. Legacy n_sites and sites_per_route keys count these input coordinates.',mean_scan_coverage=float(support.coverage.mean()),mean_sites_per_route=float(support.sites_per_route.mean()),all_one_site_scans=int(support.one_site_routes.sum()),n_scans=13,n_targets=7,source_sha256={str(p.relative_to(J)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},layout_findings=problems)
 (REC/'figure_08_sources.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
