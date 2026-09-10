#!/usr/bin/env python3
"""Per-supplementary-figure Source Data tables (SI_PLAN v2 section 5.6).

One ``source_data/curated_publication/si_S<NN>_plotted.csv`` per curated
supplementary figure, in one fixed schema:

    figure, panel, record_type, series, x_label, x, y, ci_low, ci_high, n,
    inferential_unit, interval_type, endpoint, source_table

Two kinds of row, distinguished by ``record_type`` and never mixed silently:

* ``plotted``      -- the drawn values themselves, for panels this repository
                      renders natively for the supplement;
* ``source_index`` -- for a panel pasted from a frozen source sheet, one row
                      per registered numerical source table, so a reader can
                      go from the printed panel to the file that holds its
                      numbers.  The plotted values of those panels are the
                      frozen sheets' own outputs and are not re-derived here,
                      because re-deriving them would risk changing a printed
                      number (SI_PLAN invariant 1).

Run after scripts/supplement_consolidation/build.py.
"""
from __future__ import annotations
import csv,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'scripts/supplement_consolidation'))
MAN=ROOT/'configs/supplement_consolidation/manifest.json'
OUT=ROOT/'source_data/curated_publication'
FIELDS=['figure','panel','record_type','series','x_label','x','y','ci_low','ci_high','n',
        'inferential_unit','interval_type','endpoint','source_table']

def bound_rows(figure,panel):
    """Analytic rows for the one-step bound demoted out of main Fig. 1."""
    import numpy as np
    rows=[]
    for rank,q,label in [(1,0.8,'K = 1, q = 0.8'),(2,1.0,'K = 2, q = 1')]:
        for sigma2 in np.linspace(0.0,2.0,41):
            rows.append({'figure':figure,'panel':panel,'record_type':'plotted','series':label,
                         'x_label':'noise variance sigma^2 (arbitrary units)','x':round(float(sigma2),3),
                         'y':float(q**2/(q+rank*sigma2)),'ci_low':'','ci_high':'','n':'',
                         'inferential_unit':'analytic, no data','interval_type':'none',
                         'endpoint':'2L x optimized one-step bound q^2/(q + K sigma^2)',
                         'source_table':'analytic; scripts/build_si_restored_panels.py::utility_bound'})
    return rows

def native_rows(figure,panel):
    """Rows for the merged checkpoint panel this repository draws itself."""
    import numpy as np,pandas as pd
    src=ROOT/'source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'
    data=pd.read_csv(src)
    data=data[np.isclose(data.relative_step,1e-5)]
    families=[('global_scalar_available','strict scalar'),('ancestry_available','per-neuron'),
              ('exact_transport','exact path')]
    metrics=[('gradient_cosine','gradient cosine'),
             ('norm_matched_fraction_of_exact','one-step progress')]
    rng=np.random.default_rng(0);rows=[]
    for mi,(metric,xlabel) in enumerate(metrics):
        for fi,(family,label) in enumerate(families):
            vals=data[data.feedback_family.eq(family)][metric].to_numpy()
            draws=np.random.default_rng(700+mi*10+fi).choice(vals,size=(10000,len(vals)),replace=True)
            means=draws.mean(axis=1)
            rows.append({'figure':figure,'panel':panel,'record_type':'plotted','series':label,
                         'x_label':xlabel,'x':mi*3.6+fi,'y':float(np.mean(vals)),
                         'ci_low':float(np.percentile(means,2.5)),'ci_high':float(np.percentile(means,97.5)),
                         'n':len(vals),'inferential_unit':'trained checkpoint',
                         'interval_type':'95% checkpoint bootstrap (10,000 draws)',
                         'endpoint':'valid trained checkpoint, relative step 1e-5',
                         'source_table':str(src.relative_to(ROOT))})
    return rows

def main():
    man=json.loads(MAN.read_text());OUT.mkdir(parents=True,exist_ok=True)
    written=[]
    for asset in man['assets']:
        figure=asset['figure'];rows=[]
        for panel in asset['panels']:
            letter=panel['panel']
            if panel.get('record_type')=='native supplement panel':
                rows.extend(native_rows(figure,letter) if panel['source']=='X1'
                            else bound_rows(figure,letter));continue
            sources=[q.split('journal/')[-1] for q in panel.get('numerical_source_paths',[])]
            if not sources:
                sources=[d for d in asset.get('source_data_directories',[])]
            seen=set()
            for src in sources:
                if src in seen:continue
                seen.add(src)
                rows.append({'figure':figure,'panel':letter,'record_type':'source_index','series':'',
                             'x_label':'','x':'','y':'','ci_low':'','ci_high':'','n':'',
                             'inferential_unit':'','interval_type':'','endpoint':'',
                             'source_table':src})
        number=int(figure[1:])
        dest=OUT/f'si_S{number:02d}_plotted.csv'
        with dest.open('w',newline='') as fh:
            w=csv.DictWriter(fh,fieldnames=FIELDS);w.writeheader()
            for r in rows:w.writerow(r)
        written.append({'figure':figure,'path':str(dest.relative_to(ROOT)),'rows':len(rows),
                        'id':asset['id'],'artwork':asset['path']})
        print(f"{figure:<4s} {dest.name} {len(rows)} rows")
    index=OUT/'si_manifest.json'
    index.write_text(json.dumps({'schema':'supplement-source-data/1',
        'builder':'scripts/build_supplementary_source_data.py',
        'note':('record_type=plotted rows are drawn values; record_type=source_index rows name the '
                'frozen numerical source of a pasted panel. No plotted number is re-derived here.'),
        'records':written},indent=2)+'\n')
    print('wrote',index.relative_to(ROOT))

if __name__=='__main__':main()
