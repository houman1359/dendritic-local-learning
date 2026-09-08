"""Preserve original expectations while verifying declared released hash chains."""
import hashlib,importlib.util,json
from pathlib import Path
J=Path(__file__).resolve().parents[2];OUT=J/'source_data/conductance_local_gate'
PROTOCOL_SHA256='c441998f6f91e1f8782b2918689806e119af567e479bcd5c94069dca837a2413'
FREEZE_SHA256='81568da7f4350a18c264914d1a86f7307f3809743b8c3113efa4abcc70d3bbdc'

def verify(path,original):
    path=Path(path).resolve();actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual==original:return dict(path=str(path.relative_to(J)),original_sha256=original,released_sha256=actual,reason='original bytes')
    helper=J/'code/release_noise/release_hashes.py'
    if not helper.is_file():raise ValueError('Changed canonical bytes require the reviewer release hash-chain helper')
    spec=importlib.util.spec_from_file_location('local_gate_release_hash_verifier',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.verify_released_file(path,original,journal_root=J)
    if not result['verified']:raise ValueError(f'Unverified canonical input {path}: {result["reason"]}')
    return dict(path=str(path.relative_to(J)),original_sha256=original,released_sha256=actual,reason=result['reason'])

def load_protocol():
    checks=[verify(OUT/'protocol_freeze.json',FREEZE_SHA256),verify(OUT/'protocol.json',PROTOCOL_SHA256)]
    freeze=json.loads((OUT/'protocol_freeze.json').read_text());assert freeze['protocol_sha256']==PROTOCOL_SHA256
    for relative,original in freeze['scientific_source_sha256'].items():
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid frozen relative source path')
        checks.append(verify(J/p,original))
    return json.loads((OUT/'protocol.json').read_text()),freeze,checks
