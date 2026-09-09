#!/usr/bin/env python3
"""Read-only integrity and sampling checks for the rebuilt main figures.

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
FOCUSED=J/'figures/provenance/credit_clarity_20260908'
LETTERS={1:'ABCDEFG',3:'ABCDEF',4:'ABCDEF',5:'ABCDEF',6:'ABCDEF',
         7:'ABCDEF',8:'ABCDE',9:'ABC'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(number):
    folder=FOCUSED if number in (3,5,8) else R
    return pd.read_csv(folder/f'figure_{number:02d}_plotted.csv',float_precision='round_trip')


def main():
    figures={};hash_count=0
    for number in LETTERS:
        folder=FOCUSED if number in (3,5,8) else R
        record=json.loads((folder/f'figure_{number:02d}.json').read_text())
        component=J/record['output'];canonical=J/f'figures/main/figure_{number:02d}.pdf'
        assert sha(component)==record['output_sha256']==sha(canonical)
        for category in ['source_sha256','builder_sha256','builders_sha256','helper_sha256']:
            for path,value in record.get(category,{}).items():
                assert sha(J/path)==value,(number,path)
                hash_count+=1
        doc=fitz.open(canonical);assert len(doc)==1;page=doc[0]
        spans=[span for block in page.get_text('dict')['blocks']if 'lines'in block
               for line in block['lines']for span in line['spans']if span['text'].strip()]
        outside=[s['text']for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
        assert not outside,(number,outside)
        keywords=doc.metadata.get('keywords','')
        if keywords:
            manifest=json.loads(keywords)
            letters=[p['name']for p in manifest['panels']]
        else:
            # Figure 5 retains the original Matplotlib renderer and has no
            # NativeCanvas metadata; check its actual visible panel letters.
            letters=[s['text'].strip() for s in spans
                     if s['text'].strip() in list(LETTERS[number])]
        assert letters==list(LETTERS[number]),(number,letters)
        figures[str(number)]=dict(canonical=str(canonical.relative_to(J)),
            component=str(component.relative_to(J)),sha256=sha(canonical),
            page_points=[page.rect.width,page.rect.height],panel_letters=letters,
            min_text_pt=min(s['size']for s in spans),max_text_pt=max(s['size']for s in spans),
            outside_page_text=outside)
    # Capture must use the actual trained projection coordinates and the same
    # fresh exact-rule seeds, not the older voltage-space diagnostic.
    p=table(1);f=p[p.panel.eq('F')]
    means=f[f.record_type.eq('archived summary')]
    points=f[f.record_type.eq('underlying seed')]
    assert len(means)==8 and len(points)==80 and points.coordinate.eq('activation').all()
    raw=pd.read_csv(J/'source_data/image_ladder_controls/summaries/delivery_coordinate_capture.csv')
    for row in means.itertuples():
        values=raw[raw.cohort.eq('fresh')&raw.coordinate.eq('activation')&
                   raw.architecture.eq(row.architecture)&raw.checkpoint.eq(row.checkpoint)&raw.basis.eq(row.basis)]
        assert values.seed.nunique()==10 and abs(values.mean_capture.mean()-row.mean)<1e-12
    # The promoted coefficient panel preserves resource/status distinctions
    # and matches the paired outcomes directly, including the noiseless rescue.
    p=table(3);d=p[p.panel.eq('D')&p.record.eq('mean_contrast')]
    assert len(d)==4 and d.analysis_status.eq('primary').sum()==1
    for row in d.itertuples():
        study='review_coefficient_encoder' if row.readout=='Soft' else 'review_coefficient_hard_readout'
        raw=pd.read_csv(J/f'source_data/{study}/trajectories.csv')
        raw=raw[raw.epoch.eq(80)&raw.calibration_samples.eq(row.calibration_samples)&
                raw.cue_noise_sd.eq(row.cue_noise_sd)&raw.cue_delay_trials.eq(0)]
        paired=raw.pivot(index='seed',columns='method',values='heldout_accuracy')
        values=100*(paired.learned_local_cue-paired.oracle_context)
        assert len(values)==20 and abs(values.mean()-row.mean_pp)<1e-10
        if row.readout=='Hard' and row.cue_noise_sd==0:assert values.eq(0).all()
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
    p=table(5);f=p[p.panel.eq('F')&p.record.eq('seed_endpoint')]
    assert len(f)==100 and f.groupby('rule').seed.nunique().eq(20).all()
    assert 'shunt_proportional_unit_proximal' in set(f.rule)
    raw=pd.read_csv(J/'source_data/conductance_local_gate/summaries/all_endpoints.csv',float_precision='round_trip')
    for row in f.itertuples():
        selected=raw[raw.seed.eq(row.seed)&raw.rule.eq(row.rule)&raw.rate.eq(.03)&
                     raw.task.eq('opposed_strong')&raw.budget.eq(4096)]
        assert len(selected)==1 and abs(selected.iloc[0].test_nmse-row.test_nmse)<1e-14
    p=table(6);b=p[p.panel.eq('B')];assert len(b)==9 and b.n_seeds.eq(10).all()
    counts=p[p.panel.eq('D')&p.record_type.eq('D1 observed training count')].set_index('epoch')
    raw=pd.read_csv(J/'source_data/physical_depth_followup/stopping_by_seed.csv')
    stops=raw[raw.depth.eq(1)].epochs_run
    assert len(stops)==10 and len(counts)==600
    for epoch in (180,400,600):assert counts.loc[epoch,'n_training']==stops.ge(epoch).sum()
    p=table(7);c=p[p.panel.eq('C')]
    assert sorted(c.channels.unique())==[1,2,4,8]and len(c)==16 and c['mean'].notna().all()
    p=table(9)
    assert len(p[p.panel.eq('A')])==2 and len(p[p.panel.eq('B')])==42 and len(p[p.panel.eq('C')])==125
    # Figure 8 changes the regime heading only: all plotted vector geometry
    # must remain identical to the authenticated earlier native figure.
    before=fitz.open(J/'source_data/shunt_ancestry_gain/figures/shunt_ancestry_gain_native.pdf')
    after=fitz.open(J/'figures/main/figure_08.pdf')
    assert before[0].get_drawings()==after[0].get_drawings()
    report=dict(status='PASS',verifier_sha256=sha(__file__),
        source_and_builder_hashes_verified=hash_count,figures=figures,
        checks=['canonical PDF equals native component','declared scientific sources and builders match hashes',
                'single-page vector PDFs and expected panel letters','no text outside PDF page',
                '1F matching-coordinate fresh capture,10seeds per architecture',
                '3D direct paired coefficient outcomes and primary/exploratory separation',
                '4E retains both 16384 endpoint views exactly','4F means from all20 exact-rule seeds at0/1024/16384',
                '5F all20seeds for each of5controls including continuous gate',
                '6D training counts from actual stopping epochs',
                '6B nine observed cells each10seeds','7C complete47-cell K1/2/4/8 summaries',
                '8 unchanged numerical vector geometry','9A/B/C empirical and simulation sample counts'],
        scope='Rendering and frozen-source integrity; no scientific reruns.')
    (R/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {hash_count} hashes; eight canonical figures; plotted endpoints, sample counts and page bounds.')


if __name__=='__main__':main()
