#!/usr/bin/env python3
"""Redraw only native figures from existing complete summaries; no training/summary edits."""
import json
import hashlib
from pathlib import Path
import pandas as pd
from report import OUT,figures
from portable_contract import J,load_protocol,verify

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def figure_provenance(cfg):
    """Record present render inputs separately from the immutable initial audit."""
    files=[OUT/'figures'/name for name in ['local_gate_primary.pdf','local_gate_primary.png','local_gate_all_rates.pdf','local_gate_all_rates.png','curve_band_source.csv','historical_cancellation_source.csv']]
    inputs=list(sorted((OUT/'summaries').glob('*.csv')))+[OUT/'protocol.json',OUT/'protocol_freeze.json',J/'source_data/conductance_credit_demand/opponent/summaries/context_gradient_summary.csv']
    code=[Path(__file__),Path(__file__).with_name('report.py'),Path(__file__).with_name('portable_contract.py'),Path(__file__).with_name('model.py'),J/'scripts/journal_style.py']
    def hashes(paths):return {str(p.relative_to(J)):digest(p) for p in paths}
    record=dict(status='PASS',scope='Current figure-only rendering; frozen protocol and numerical summary bytes unchanged. Initial numerical audit retains its historical render hashes.',protocol_sha256=digest(OUT/'protocol.json'),fresh_seeds=cfg['fresh_seeds'],historical_seeds=list(range(2101,2121)),figure_sha256=hashes(files),input_sha256=hashes(inputs),code_sha256=hashes(code),panel_sources={
        'main_5A':{'type':'model schematic','sources':['scripts/conductance_local_gate/model.py','scripts/conductance_local_gate/report.py'],'scope':'Local distal selector and unit proximal credit; no experimental outcomes.'},
        'main_5B':{'type':'nominal teacher tuning','sources':['scripts/conductance_local_gate/model.py','scripts/conductance_local_gate/report.py'],'scope':'Nominal unperturbed tuning curves, not fitted student results.'},
        'main_5C_D':{'type':'fresh cohort','sources':['source_data/conductance_local_gate/summaries/all_curves.csv','source_data/conductance_local_gate/figures/curve_band_source.csv'],'scope':'Adam 0.03; four prespecified rules; all twenty fresh seeds. Display ribbons are descriptive 20000-resample seed-bootstrap intervals, separate RNG 2026090830.'},
        'main_5E':{'type':'fresh cohort','sources':['source_data/conductance_local_gate/summaries/all_endpoints.csv','source_data/conductance_local_gate/summaries/paired_contrasts.csv'],'scope':'Paired task-by-rule interaction at primary and extended validation windows.'},
        'main_5F':{'type':'fresh cohort','sources':['source_data/conductance_local_gate/summaries/all_endpoints.csv','source_data/conductance_local_gate/summaries/condition_means.csv'],'scope':'Primary-rate opposed endpoints: exact, local distal gate, swapped gate and proximal-gating failure control.'},
        'supp_53A_D':{'type':'fresh cohort','sources':['source_data/conductance_local_gate/summaries/condition_means.csv'],'scope':'All nine rules, three rates, two tasks and two validation windows; 1080 trajectories and 2160 endpoint views.'},
        'supp_53E':{'type':'historical cohort','sources':['source_data/conductance_credit_demand/opponent/summaries/context_gradient_summary.csv','source_data/conductance_local_gate/figures/historical_cancellation_source.csv'],'scope':'Earlier seeds 2101–2120; opposed task; calibrated-broadcast source weights; exact and calibrated evaluations; distal parameter gradients at initial and extended-best states. Does not evaluate new local-gate fits.'}})
    record['source_sha256']={**record['input_sha256'],**record['code_sha256']}
    (OUT/'figure_provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')

def main():
    cfg,freeze,_=load_protocol();audit=json.loads((OUT/'summaries/completeness_audit.json').read_text());assert audit['status']=='PASS'
    folder=OUT/'summaries'
    for relative,original in audit['summary_sha256'].items():verify(OUT/relative,original)
    frames={k:pd.read_csv(folder/f'all_{k}.csv',float_precision='round_trip') for k in ['endpoints','curves','diagnostics']}
    means=pd.read_csv(folder/'condition_means.csv',float_precision='round_trip');contrasts=pd.read_csv(folder/'paired_contrasts.csv',float_precision='round_trip')
    figures(cfg,frames,means,contrasts)
    figure_provenance(cfg)
    print('Native figures redrawn; experiment protocol and numerical summary files unchanged.')
if __name__=='__main__':main()
