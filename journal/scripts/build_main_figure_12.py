#!/usr/bin/env python3
"""Substantive morphology and learning result, staged as native component12.

The original prospective component11 remains reproducible and is retained in
Supplementary Figure35. Canonical publication numbering is set by assembly.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from figure_canvas import NativeCanvas,Margins,PT_SMALL,PT_ANNOT,PT_LABEL,LW_DATA,LW_REF
from journal_style import COLORS
from build_morphology_followup_figures import FC
from build_boolean_morphology_figures import grouping_schematic
from build_morphology_bridge_figures import pipeline_structure,pipeline_credit
ROOT=Path(__file__).resolve().parents[1]
STRUCT=ROOT/'source_data/morphology_structure'
CALIB=ROOT/'source_data/morphology_calibration'
COND=ROOT/'source_data/morphology_conductance'

def schematic(ax):
 grouping_schematic(ax)

def cut_panel(ax):
 end=pd.read_csv(STRUCT/'candidate_outcomes.csv')
 for family,color,label in [('quadratic_matching',FC['quadratic_matching'],'matching'),('quartic_partition',FC['quartic_partition'],'quartic')]:
  z=end[end.family.eq(family)].copy();z['x']=z.centered_cut_bound.round(7);z['y']=z.normalized_mse.round(7)
  z=z.groupby(['x','y']).size().reset_index(name='n')
  ax.scatter(z.x,z.y,s=8+3*np.sqrt(z.n),color=color,marker='o' if family=='quadratic_matching' else '^',alpha=.7,label=label,lw=0,zorder=3)
 ax.plot([0,.8],[0,.8],ls='--',color='#98A2A9',lw=LW_REF)
 ax.set_xlim(-.04,.82);ax.set_ylim(-.04,.91);ax.set_xticks([0,.25,.5,.75]);ax.set_yticks([0,.25,.5,.75])
 ax.set_xlabel('centered-cut bound (NMSE)');ax.set_ylabel('best fitted NMSE')
 ax.text(.03,.98,r'Matching and quartic: $\mathbb{E}[\nabla_x f\nabla_x f^\mathsf{T}]=I_8/4$',transform=ax.transAxes,va='top',fontsize=PT_SMALL)
 ax.legend(loc='lower right',frameon=False,fontsize=PT_SMALL)

def calibration_primary(ax):
 source=pd.read_csv(CALIB/'policy_summary.csv')
 z=source[source.family.eq('all')&source.calibration_rows.eq(256)&source.calibration_noise_sd.eq(.5)].set_index('policy')
 keys=['estimated_cut','development_best_fixed','two_sweep_pilot','uniform_random_expectation','oracle_full_target_cut']
 labels=['estimated interactions','fixed / estimated-rank','two-sweep fitting pilot','random expectation','true-interaction oracle']
 colors=[COLORS['shunting'],COLORS['point_mlp'],COLORS['local'],COLORS['mute'],COLORS['oracle']]
 for y,key,label,color in zip(range(5),keys,labels,colors):
  row=z.loc[key]
  ax.errorbar(row.mean_regret,y,xerr=[[row.mean_regret-row.regret_ci_low],[row.regret_ci_high-row.mean_regret]],fmt='o',ms=3.5,color=color,lw=LW_DATA,capsize=1.8)
  ax.text(.02,y-.30,label,transform=ax.get_yaxis_transform(),fontsize=PT_SMALL,color=color)
 ax.set_yticks([]);ax.set_ylim(4.4,-.8);ax.set_xlim(-.006,.23);ax.set_xlabel('regret within twelve candidates')

def adaptive_panel(ax):
 frame=pd.read_csv(CALIB/'figure_absolute_error_summary.csv')
 families=['matching','quartet','nested_prefix','random_interactions']
 for j,(policy,color,label) in enumerate([('estimated_adaptive_dp','#278365','estimated tree'),('oracle_best_trained_menu','#7D858A','best of twelve (oracle)'),('oracle_adaptive_dp','#8055AD','true-target tree (oracle)')]):
  z=frame[frame.calibration_rows.eq(256)&frame.calibration_noise_sd.eq(.5)&frame.policy.eq(policy)&frame.family.isin(families)].set_index('family').loc[families]
  ax.errorbar(np.arange(4)+(j-1)*.11,z.mean_test_nmse,yerr=[z.mean_test_nmse-z.ci95_low,z.ci95_high-z.mean_test_nmse],fmt='o',ms=3.1,color=color,lw=LW_DATA,capsize=1.4,label=label)
 ax.set_xticks(range(4),['matching','quartic','nested','random']);ax.set_yscale('log');ax.set_ylim(2e-4,3);ax.set_ylabel('clean-test NMSE')
 ax.legend(frameon=False,fontsize=PT_SMALL,loc='center left',bbox_to_anchor=(.015,.5))

def conductance_panel(ax):
 source=COND/'summaries/fresh'
 summary=pd.read_csv(source/'learning_summary.csv')
 end=pd.read_csv(source/'all_learning_curves.csv')
 step=end.step.max();end=end[end.step.eq(step)&end.optimizer.eq('adam')&end.credit_rule.eq('exact_path')]
 for x,compatible in enumerate([True,False]):
  values=end[end.compatible.eq(compatible)].groupby('seed').test_nmse.mean().sort_index()
  assert len(values)==20
  row=summary[summary.step.eq(step)&summary.optimizer.eq('adam')&summary.credit_rule.eq('exact_path')&summary.compatible.eq(compatible)].iloc[0]
  color=COLORS['shunting'] if compatible else COLORS['local']
  ax.scatter(x+np.linspace(-.12,.12,20),values,s=9,color=color,alpha=.4,zorder=3)
  ax.errorbar(x,row.mean_test_nmse,yerr=[[row.mean_test_nmse-row.ci95_low],[row.ci95_high-row.mean_test_nmse]],fmt='D',ms=4,color=color,lw=LW_DATA,capsize=2,zorder=4)
 upper=float(end.groupby(['compatible','seed']).test_nmse.mean().max())*1.40
 ax.set_xticks([0,1],['compatible\ngrouping','incompatible\ngroupings']);ax.set_xlim(-.4,1.4);ax.set_ylim(-.001,upper);ax.set_ylabel('test NMSE')
 ax.text(.04,.98,'Positive conductances; same physical shape',transform=ax.transAxes,va='top',fontsize=PT_SMALL)
 ax.text(.04,.86,'Adam; exact path credit; 20 seed blocks',transform=ax.transAxes,va='top',fontsize=PT_SMALL)

def build():
 canvas=NativeCanvas(495/72,3,row_weights=[126,135,135],hgutter_pt=37,vgutter_pt=45,margins=Margins(left=43,right=13,top=22,bottom=38))
 a=canvas.panel('A',0,0,6,schematic=True,title='Multi-affine model: input grouping')
 b=canvas.panel('B',0,6,6,title='Equal spectrum, different compatibility')
 c=canvas.panel('C',1,0,6,title='Selecting from 256 noisy labels',grid='x')
 d=canvas.panel('D',1,6,6,title='Estimated trees train with exact credit',grid='y')
 e=canvas.panel('E',2,0,6,title='Estimated trees: task-dependent credit',grid='y')
 f=canvas.panel('F',2,6,6,title='Conductance: grouping controls learning',grid='y')
 schematic(a);cut_panel(b);calibration_primary(c);pipeline_structure(d)
 pipeline_credit(e)
 conductance_panel(f)
 from journal_style import style_direct_color_labels
 style_direct_color_labels(canvas.fig)
 output=canvas.save(ROOT/'figures/components/main_figure_12_native.pdf',name='main_figure_12_native')
 sources=[ROOT/'source_data/boolean_theory/main6a_grouping.csv',
          ROOT/'source_data/boolean_theory/gate_credit_fields.csv',STRUCT/'candidate_outcomes.csv',
          CALIB/'policy_summary.csv',CALIB/'primary_contrasts.csv',CALIB/'end_to_end/summary.csv',
          CALIB/'end_to_end/contrasts.csv',CALIB/'end_to_end/protocol.json',
          COND/'summaries/fresh/learning_summary.csv',COND/'summaries/fresh/all_learning_curves.csv',
          COND/'summaries/fresh/paired_contrasts.csv']
 builders=[Path(__file__).resolve(),ROOT/'scripts/build_morphology_bridge_figures.py',ROOT/'scripts/build_morphology_followup_figures.py',ROOT/'scripts/build_boolean_morphology_figures.py']
 record={'canonical_figure':6,'native_component':12,'panels':'A-F',
         'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         'builders':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in builders},
         'scope':'A canonical Boolean gate illustration with all-coefficient regression obstruction; B exact-target multi-affine class; C finite-calibration cohort; D-E separate end-to-end cohort with frozen per-rule Adam rates; F fixed-shape conductance grouping.'}
 (ROOT/'figures/components/main_figure_12_native.provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
 return output

if __name__=='__main__':build()
