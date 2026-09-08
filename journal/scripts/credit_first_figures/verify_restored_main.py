#!/usr/bin/env python3
"""Read-only integrity and sampling checks for the five restored main figures.

The only output is a rendering verification record under figures/provenance.
This does not run training or mutate scientific Source Data.
"""
from pathlib import Path
import hashlib
import json

import fitz
import numpy as np
import pandas as pd

J=Path(__file__).resolve().parents[2]
R=J/'figures/provenance/structure_restoration_20260908'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(number):
    return pd.read_csv(R/f'figure_{number:02d}_plotted.csv',float_precision='round_trip')


def main():
    figures={};hash_count=0
    for number in [1,4,6,7,9]:
        record=json.loads((R/f'figure_{number:02d}.json').read_text())
        component=J/record['output'];canonical=J/f'figures/main/figure_{number:02d}.pdf'
        assert sha(component)==record['output_sha256']==sha(canonical)
        for category in ['source_sha256','builder_sha256','helper_sha256']:
            for path,value in record.get(category,{}).items():
                assert sha(J/path)==value,(number,path)
                hash_count+=1
        doc=fitz.open(canonical);assert len(doc)==1;page=doc[0]
        spans=[span for block in page.get_text('dict')['blocks']if 'lines'in block
               for line in block['lines']for span in line['spans']if span['text'].strip()]
        outside=[s['text']for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
        assert not outside,(number,outside)
        manifest=json.loads(doc.metadata['keywords'])
        letters=[p['name']for p in manifest['panels']]
        assert letters==list('ABC'if number==9 else'ABCDEF')
        figures[str(number)]=dict(canonical=str(canonical.relative_to(J)),
            component=str(component.relative_to(J)),sha256=sha(canonical),
            page_points=[page.rect.width,page.rect.height],panel_letters=letters,
            min_text_pt=min(s['size']for s in spans),max_text_pt=max(s['size']for s in spans),
            outside_page_text=outside)
    p=table(4);e=p[p.panel.eq('E')&p.budget.eq(16384)];assert len(e)==2
    s=pd.read_csv(J/'source_data/credit_rule_extension/summaries/paired_contrasts.csv',float_precision='round_trip')
    s=s[s.budget.eq(16384)&s.task.eq('quartet_minus_matching')&s.optimizer.eq('adam')&
        s.rate_view.eq('selected_rate')&s.metric.eq('test_nmse')&
        s.contrast.eq('calibrated_broadcast minus exact interaction')]
    assert np.array_equal(e.sort_values('endpoint')['mean'],s.sort_values('endpoint')['mean'])
    f=p[p.panel.eq('F')];assert len(f)==9 and sorted(f.step.unique())==[0,1024,16384]
    raw=pd.read_csv(J/'source_data/credit_rule_extension/summaries/all_diagnostics.csv',float_precision='round_trip')
    raw=raw[raw.rule.eq('exact')&raw.optimizer.eq('adam')&raw.selected_rate&raw.state.eq('own_checkpoint')]
    for row in f.itertuples():
        v=raw[raw.task.eq(row.task)&raw.step.eq(row.step)].path_best_rank_one_capture
        assert len(v)==20 and abs(v.mean()-row.mean)<1e-14
    p=table(6);b=p[p.panel.eq('B')];assert len(b)==9 and b.n_seeds.eq(10).all()
    p=table(7);c=p[p.panel.eq('C')]
    assert sorted(c.channels.unique())==[1,2,4,8]and len(c)==16 and c['mean'].notna().all()
    p=table(9)
    assert len(p[p.panel.eq('A')])==2 and len(p[p.panel.eq('B')])==42 and len(p[p.panel.eq('C')])==125
    report=dict(status='PASS',verifier_sha256=sha(__file__),
        source_and_builder_hashes_verified=hash_count,figures=figures,
        checks=['canonical PDF equals native component','declared scientific sources and builders match hashes',
                'single-page vector PDFs and expected panel letters','no text outside PDF page',
                '4E retains both 16384 endpoint views exactly','4F means from all20 exact-rule seeds at0/1024/16384',
                '6B nine observed cells each10seeds','7C complete47-cell K1/2/4/8 summaries',
                '9A/B/C empirical and simulation sample counts'],
        scope='Rendering and frozen-source integrity; no scientific reruns.')
    (R/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {hash_count} hashes; five canonical figures; plotted endpoints, sample counts and page bounds.')


if __name__=='__main__':main()
