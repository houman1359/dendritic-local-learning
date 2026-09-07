#!/usr/bin/env python3
"""Report the frozen posthoc grid and its separately frozen added trigger."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;J=HERE.parents[1]
sys.path.insert(0,str(HERE.parent))
from figure_canvas import NativeCanvas,Margins,COLORS,PT_LEGEND,LW_DATA,LW_REF
import expanded_rates as study

def main(plot_only=False):
    # Plotting and posthoc arithmetic do not invoke the original training
    # environment guard. Scientific input identities are checked separately.
    out=study.OUT;record=json.loads((out/'protocol_freeze.json').read_text())
    import importlib.util
    spec=importlib.util.spec_from_file_location('_expanded_rate_release_hashes',J/'code/release_noise/release_hashes.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    def verify(path,digest):
        verdict=helper.verify_released_file(path,digest,journal_root=J)
        if not verdict['verified']:raise ValueError(f'Unverified expanded-grid input: {path}')
    verify(HERE/'expanded_rates.py',record['source_sha256'])
    for seed in record['development_seeds']:
        audit=json.loads((out/'runs'/f'seed_{seed}_audit.json').read_text())
        verify(out/'protocol_freeze.json',audit['freeze_sha256'])
        for relative,digest in audit['files_sha256'].items():verify(out/'runs'/relative,digest)
    addendum=json.loads((out/'validation_trigger_addendum.json').read_text())
    verify(out/'protocol_freeze.json',addendum['original_grid_protocol_sha256'])
    table=pd.read_csv(out/'rate_comparison.csv')
    endpoints=pd.concat([pd.read_csv(out/'runs'/f'seed_{seed}_endpoints.csv') for seed in record['development_seeds']],ignore_index=True)
    reproduced=endpoints.groupby(['task','optimizer','rule','rate'],as_index=False).validation_nmse.mean()
    compared=table.merge(reproduced,on=['task','optimizer','rule','rate'],suffixes=('_table','_recomputed'),validate='one_to_one')
    assert len(endpoints)==288 and len(compared)==96
    np.testing.assert_allclose(compared.validation_nmse_table,compared.validation_nmse_recomputed,atol=1e-15,rtol=1e-13)
    original=json.loads((out/'selection_review.json').read_text())
    checks=[]
    for opt,rate in [('adam',.03),('sgd',.3)]:
        for rule in ['unit_broadcast','calibrated_broadcast']:
            q=table[(table.task=='opposed_strong')&(table.optimizer==opt)&(table.rule==rule)]
            old=float(q[q.rate==rate].validation_nmse.iloc[0]);winner=q.loc[q.validation_nmse.idxmin()];new=float(winner.validation_nmse)
            absolute=old-new;relative=absolute/max(old,1e-30)
            checks.append(dict(optimizer=opt,rule=rule,actual_original_rate=rate,actual_original_validation_nmse=old,best_rate=float(winner.rate),best_validation_nmse=new,absolute_improvement=absolute,relative_improvement=relative,material=absolute>=.01 and relative>=.1))
    verdict=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),original_trigger_review_sha256=study.sha(out/'selection_review.json'),pre_outcome_addendum_sha256=study.sha(out/'validation_trigger_addendum.json'),original_expanded_vs_original_grid_trigger=original['trigger_followup'],additional_checks=checks,trigger_followup=original['trigger_followup'] or any(r['material'] for r in checks),n_fits=original['n_fits'],outcome='No additional fresh-seed fits are triggered; every rate and outcome remains available.',selection_uses='Mean development validation NMSE, never test NMSE.')
    if verdict['trigger_followup']:verdict['outcome']='Follow-up is required under the frozen criteria.'
    if plot_only:
        retained=json.loads((out/'complete_trigger_review.json').read_text())
        assert retained['additional_checks']==checks
        assert retained['trigger_followup']==verdict['trigger_followup']
        assert retained['n_fits']==verdict['n_fits']
        verdict=retained
    else:
        study.write(out/'complete_trigger_review.json',verdict)
    colors=dict(exact=COLORS['bp'],ancestry_three_oracle=COLORS['oracle'],calibrated_broadcast=COLORS['additive'],unit_broadcast=COLORS['scalar'])
    labels=dict(exact='Exact path',ancestry_three_oracle='Three profiles (oracle)',calibrated_broadcast='Initial profile',unit_broadcast='Unit broadcast')
    styles=dict(exact='-',ancestry_three_oracle='-.',calibrated_broadcast=':',unit_broadcast='--')
    cv=NativeCanvas(422/72,2,row_weights=[163,163],hgutter_pt=36,vgutter_pt=55,margins=Margins(left=60,right=20,top=26,bottom=42));rows=[]
    for panel,row,col,opt,task in [('A',0,0,'adam','opposed_strong'),('B',0,6,'sgd','opposed_strong'),('C',1,0,'adam','aligned_strong'),('D',1,6,'sgd','aligned_strong')]:
        title=f'{"Adam" if opt=="adam" else "SGD"}: {"opposed" if task=="opposed_strong" else "aligned"} tuning'
        ax=cv.panel(panel,row,col,6,title=title,grid='y')
        for rule in ['exact','ancestry_three_oracle','unit_broadcast','calibrated_broadcast']:
            d=table[(table.optimizer==opt)&(table.task==task)&(table.rule==rule)].sort_values('rate')
            ax.plot(d.rate,d.validation_nmse,color=colors[rule],ls=styles[rule],marker='o' if rule!='calibrated_broadcast' else 's',mfc='none' if rule=='calibrated_broadcast' else colors[rule],ms=3.5,lw=LW_DATA,label=labels[rule])
            for r in d.to_dict('records'):rows.append(dict(panel=panel,**r))
        old=.03 if opt=='adam' else .3
        ax.axvline(old,color=COLORS['mute'],lw=LW_REF,ls=':')
        ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(5e-8,2)
        rates=record['rates'][opt];ax.set_xticks(rates,[f'{v:g}' for v in rates]);ax.set_yticks([1e-7,1e-5,1e-3,.1,1],['10⁻⁷','10⁻⁵','10⁻³','0.1','1']);ax.minorticks_off()
        ax.set_xlabel('Learning rate');ax.set_ylabel('Development validation NMSE')
        if panel=='A':ax.legend(frameon=False,fontsize=PT_LEGEND,loc='center left',handlelength=1.6)
    pd.DataFrame(rows).to_csv(out/'figure_expanded_rates_source.csv',index=False)
    cv.save(out/'figure_expanded_rates.pdf')
    provenance=dict(figure='Supplementary Fig. S52',builder=str(Path(__file__).relative_to(J)),builder_sha256=study.sha(Path(__file__)),source_csv='figure_expanded_rates_source.csv',source_sha256=study.sha(out/'figure_expanded_rates_source.csv'),inputs={name:study.sha(out/name) for name in ['protocol_freeze.json','validation_trigger_addendum.json','all_endpoints.csv','rate_comparison.csv','selected_rates.csv','complete_trigger_review.json']},pdf_sha256=study.sha(out/'figure_expanded_rates.pdf'),n_fit_endpoints=288,n_development_seeds=3,validation_selection_only=True,plot_only=plot_only,geometry_points=[518.4,422],overlap_audit='See native figure audit; legend occupies the empty central region.')
    study.write(out/'figure_expanded_rates_provenance.json',provenance)
    print(json.dumps(verdict,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plot-only',action='store_true',help='Read retained decisions without rewriting any experimental input or audit; update figure artifacts only.');args=parser.parse_args();main(plot_only=args.plot_only)
