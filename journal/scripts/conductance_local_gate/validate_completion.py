#!/usr/bin/env python3
"""Verify the complete frozen cohort and current figures without numerical reruns."""
import argparse,csv,hashlib,json
from pathlib import Path
from portable_contract import J,OUT,load_protocol,verify

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def build_gate():
    cfg,freeze,_=load_protocol();initial=read(OUT/'summaries/completeness_audit.json')
    assert initial['status']=='PASS'
    for relative,original in initial['scientific_input_sha256'].items():verify(OUT/relative,original)
    for relative,original in initial['summary_sha256'].items():verify(OUT/relative,original)
    # The first render audit remains historical. Current render hashes are separate.
    prov=read(OUT/'figure_provenance.json');assert prov['status']=='PASS'
    for key in ['source_sha256','figure_sha256']:
        for relative,expected in prov[key].items():verify(J/relative,expected)
    with (OUT/'summaries/all_endpoints.csv').open() as handle:ep=list(csv.DictReader(handle))
    with (OUT/'summaries/all_curves.csv').open() as handle:curves=list(csv.DictReader(handle))
    keys={(r['seed'],r['task'],r['budget'],r['rate'],r['rule']) for r in ep}
    assert len(ep)==len(keys)==2160 and len(curves)==12960
    assert {int(r['seed']) for r in ep}==set(cfg['fresh_seeds'])
    assert len({(r['seed'],r['task'],r['rate'],r['rule']) for r in ep})==1080
    counts={}
    for row in ep:
        key=(row['task'],row['budget'],row['rate'],row['rule']);counts[key]=counts.get(key,0)+1
    assert len(counts)==108 and set(counts.values())=={20}
    replay=read(OUT/'historical_replay/report.json');assert replay['status']=='PASS' and len(replay['checks'])==8 and all(x['passed'] for x in replay['checks'])
    assert '10 passed in 2.61s' in (OUT/'canary_45289328.log').read_text()
    portable=[]
    for rule in ['hard_distal','two_leaf_oracle']:
        folder=OUT/'portable_validation'/rule;report=read(folder/'replay_report.json');check=report['canonical_comparison']
        assert check['within_roundoff_tolerance'] and check['canonical_parameters_max_abs_difference']==0 and check['canonical_endpoint_nmse_max_abs_difference']==0
        for relative,expected in report['files_sha256'].items():verify(folder/relative,expected)
        portable.append(dict(path=str((folder/'replay_report.json').relative_to(J)),sha256=sha(folder/'replay_report.json'),rule=report['rule'],canonical_parameters_max_abs_difference=0,canonical_endpoint_nmse_max_abs_difference=0))
    evidence=[OUT/'summaries/completeness_audit.json',OUT/'historical_replay/report.json',OUT/'canary_45289328.log',OUT/'figure_provenance.json',OUT/'protocol.json',OUT/'protocol_freeze.json',Path(__file__)]
    evidence += sorted((OUT/'portable_validation').glob('*/*'))
    return dict(status='PASS',scope='Complete fresh cohort, unchanged numerical summaries, historical replay and current figures verified. Original completeness_audit.json figure hashes describe the initial render only.',protocol_sha256=freeze['protocol_sha256'],protocol_commit='c6d3b2b09d24221de87bad18b445f0060a6d238f',transitive_dependency_commit='37dbfe9',fresh_seeds=cfg['fresh_seeds'],n_trajectories=1080,n_endpoint_views=len(ep),n_curve_rows=len(curves),scientific_input_sha256=initial['scientific_input_sha256'],summary_sha256=initial['summary_sha256'],figure_provenance_sha256=sha(OUT/'figure_provenance.json'),source_sha256={str(p.relative_to(J)):sha(p) for p in evidence},validation=dict(mechanism_tests_passed=10,historical_4096_update_comparisons_passed=8,portable_4096_update_replays=portable,scope='Replays validate implementation and portability; they are not new independent scientific observations.'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true',help='Verify existing gate and dependencies without writing');args=ap.parse_args();gate=build_gate()
    if args.check:
        stored=read(OUT/'completion_gate.json');assert stored['status']=='PASS'
        for relative,expected in stored['source_sha256'].items():verify(J/relative,expected)
        assert stored['n_trajectories']==gate['n_trajectories'] and stored['n_endpoint_views']==gate['n_endpoint_views'] and stored['n_curve_rows']==gate['n_curve_rows']
    else:(OUT/'completion_gate.json').write_text(json.dumps(gate,indent=2,sort_keys=True)+'\n')
    print('PASS: 20 fresh seeds, 1080 trajectories, 2160 endpoint views; current figures and all retained source hashes verified.')
if __name__=='__main__':main()
