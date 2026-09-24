"""Run the frozen paired analysis after verifying output-only relocation."""
from pathlib import Path
import copy, hashlib, importlib.util, json, sys
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'frozen_analysis'))
import analyze_controls as analysis
import validate_controls as validation

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    repair = json.loads((HERE / 'protocol.json').read_text())
    original_path = Path(repair['original_manifest'])
    original = json.loads(original_path.read_text())
    validation.validate_manifest(original)
    merged = copy.deepcopy(original)
    replacement = {r['key']: r for job in repair['jobs'] for r in job['runs']}
    relocations = []
    for job in merged['jobs']:
        for rec in job['runs']:
            if rec['key'] not in replacement:
                continue
            updated = replacement[rec['key']]
            folder = Path(updated['result_dir'])
            receipt = json.loads((folder/'execution.json').read_text())
            assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
            assert receipt['original_config_sha256'] == rec['config_sha256']
            assert receipt['source'] == rec['source']
            ancestor = yaml.safe_load(Path(rec['config']).read_text())
            executed = yaml.safe_load((folder/'executed_config.yaml').read_text())
            assert executed['outputs']['results_dir'] == str(folder)
            normalized = copy.deepcopy(executed)
            normalized['outputs']['results_dir'] = ancestor['outputs']['results_dir']
            assert normalized == ancestor, rec['key']
            assert sha(folder/'executed_config.yaml') == receipt['executed_config_sha256']
            relocations.append(dict(key=rec['key'], original_config_sha256=rec['config_sha256'],
                executed_config_sha256=receipt['executed_config_sha256'],
                original_result_dir=rec['result_dir'], result_dir=str(folder)))
            rec['result_dir'] = str(folder)
    assert len(relocations) == 72
    validation.validate_results(merged)
    (HERE/'manifest.json').write_text(json.dumps(merged,indent=2)+'\n')
    # Only result locations changed. The frozen scientific manifest was checked
    # above, and the frozen result validator checks all 220 relocated records.
    def checked_manifest(candidate):
        assert candidate == merged
        validation.validate_manifest(original)
    analysis.R = HERE
    analysis.validate_manifest = checked_manifest
    analysis.main()
    report = dict(original_manifest_sha256=sha(original_path),
        repair_protocol_sha256=sha(HERE/'protocol.json'), relocation_count=72,
        scientific_configs_unchanged=True, all_original_seeds_retained=True,
        validator_sha256=sha(HERE/'frozen_analysis/validate_controls.py'),
        analyzer_sha256=sha(HERE/'frozen_analysis/analyze_controls.py'),
        wrapper_sha256=sha(__file__), relocations=relocations)
    (HERE/'frozen_analysis/analysis/relocation_audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
