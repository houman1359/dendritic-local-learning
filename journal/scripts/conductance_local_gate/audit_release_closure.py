#!/usr/bin/env python3
"""Check every gate-verifier input against Source Data and software selection."""
import argparse,json,sys
from pathlib import Path
J=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(J/'scripts'))
import build_nature_source_data as source_data
import build_software_release as software

def audit():
    root=J/'source_data/conductance_local_gate';required={}
    def add(path,why):required.setdefault(path,set()).add(why)
    for name in ['protocol.json','protocol_freeze.json','summaries/completeness_audit.json','figure_provenance.json','completion_gate.json','historical_replay/report.json','canary_45289328.log']:
        add('source_data/conductance_local_gate/'+name,'direct verifier read')
    freeze=json.loads((root/'protocol_freeze.json').read_text())
    for path in freeze['scientific_source_sha256']:add(path,'frozen source')
    initial=json.loads((root/'summaries/completeness_audit.json').read_text())
    for key in ['scientific_input_sha256','summary_sha256']:
        for path in initial[key]:add('source_data/conductance_local_gate/'+path,key)
    for name in ['figure_provenance.json','completion_gate.json']:
        record=json.loads((root/name).read_text())
        for key in ['source_sha256','figure_sha256']:
            for path in record.get(key,{}):add(path,name+' '+key)
    for name in ['hard_distal','two_leaf_oracle']:
        path=root/'portable_validation'/name/'replay_report.json';record=json.loads(path.read_text());add(path.relative_to(J).as_posix(),'portable evidence')
        for relative in record['files_sha256']:add((path.parent/relative).relative_to(J).as_posix(),'portable evidence')
    for path in ['scripts/conductance_local_gate/portable_contract.py','scripts/conductance_local_gate/portable_replay.py','scripts/conductance_local_gate/report.py','scripts/conductance_local_gate/model.py','scripts/conductance_local_gate/run.py','scripts/journal_style.py','code/release_noise/release_hashes.py']:
        add(path,'import/helper')
    inventory={item.source:item.destination for item in source_data.FILES}
    records=[]
    for path,reasons in sorted(required.items()):
        selected=software.repository_file_allowed(Path('journal')/path,'paper')
        record=dict(path=path,required_by=sorted(reasons),exists=(J/path).is_file(),source_data_destination=inventory.get(path),software_allowlisted=selected)
        record['releasable']=bool(record['exists'] and (path in inventory or selected));records.append(record)
    missing=[row['path'] for row in records if not row['releasable']]
    return dict(status='PASS' if not missing else 'FAIL',scope='Input-closure audit of the unchanged gate verifier against the actual Source Data inventory and software allowlist. This verifies selection, not archive restoration or numerical reproduction.',n_required_paths=len(records),missing_paths=missing,records=records)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);args=ap.parse_args();result=audit()
    if args.output:args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
